"""Real FR3 PC-ACT evaluation with live 40-sensor histories and independent intrusion guard.

The policy has not been trained to avoid obstacles. The guard detects novel near
returns against the recorded background; it is a heuristic abort, not learned
avoidance, a robot-clearance estimate, or a certified collision-prevention system.
"""
from collections import deque
import hashlib
import csv
import json
import os
from pathlib import Path
import time
import threading

import h5py
import numpy as np

import eval_act_real as act
import record_dataset as base
from sensor_layout import sim_name
from act_model.sensor_order import HYBRID_SKIN_SENSOR_ORDER

ROOT=Path(__file__).resolve().parent
DEFAULT_MODEL=ROOT/'models/pc_act_test_20'
NAMES=list(HYBRID_SKIN_SENSOR_ORDER)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class CausalSkinHistory:
    def __init__(self):self.frames=deque(maxlen=8)

    def append(self,depth_m):
        depth_m=np.asarray(depth_m,dtype=np.float32)
        if depth_m.shape!=(40,8,8) or not np.isfinite(depth_m).all():
            raise ValueError('expected finite 40x8x8 depth in meters')
        self.frames.append(depth_m.copy())
        frames=[self.frames[0]]*(8-len(self.frames))+list(self.frames)
        return np.repeat(np.stack(frames).transpose(1,0,2,3),4,axis=1).copy()


class PCInference(act.ACTInference):
    def __init__(self,args):
        from act_model.surface_geometry import SurfaceGeometryEncoder
        import torch
        cfg=json.loads((args.ckpt_dir/'prox_config.json').read_text())
        if cfg.get('sensor_order')!=NAMES or cfg.get('prox_policy_tap')!='readout' or cfg.get('prox_feat_dim')!=128:
            raise ValueError('checkpoint does not match the trained 40-sensor / 128D readout contract')
        encoder_name={'policy_best.ckpt':'prox_encoder_best.pt','policy_last.ckpt':'prox_encoder_last.pt'}.get(args.ckpt_name)
        if encoder_name is None:raise ValueError('use a best or last policy and its matching encoder')
        pair_path=args.ckpt_dir/'policy_pair.json'
        if not pair_path.exists():raise ValueError('deploy with export_pc_act_model.py first; missing checkpoint pair manifest')
        pair=json.loads(pair_path.read_text())
        if pair['policy_file']!=args.ckpt_name or pair['encoder_file']!=encoder_name:
            raise ValueError('policy/encoder filenames do not match the deployment pair')
        for filename,key in [(args.ckpt_name,'policy_sha256'),(encoder_name,'encoder_sha256')]:
            if sha256(args.ckpt_dir/filename)!=pair[key]:raise ValueError(f'checkpoint pair checksum mismatch: {filename}')
        payload=torch.load(args.ckpt_dir/encoder_name,map_location='cpu',weights_only=True)
        if payload.get('epoch')!=pair['epoch']:raise ValueError('encoder epoch differs from paired policy metadata')
        encoder=SurfaceGeometryEncoder(kind='embedding',checkpoint=args.ckpt_dir/encoder_name,
                                       device='cpu',frozen=True,policy_tap='readout')
        super().__init__(args,proximity_encoder=encoder)
        if int(self.config.get('n_proximity_sensors',0))!=40 or int(self.config.get('prox_feat_dim',0))!=128:
            raise ValueError('policy dimensions do not match proximity encoder')
        self.pair=pair
        print(f'PC-ACT: paired epoch {pair["epoch"]}; causal 8-tick history; policy sensor mode={args.proximity_mode}')


class ProximityAbort(RuntimeError):pass


class LiveSkinContext:
    def __init__(self,args,listener=None):
        self.args,self.listener=args,listener
        self.history=CausalSkinHistory()
        self.lock=threading.RLock()
        self.events=[];self.trigger=None;self.last_snapshot=None
        self.min_raw=float('inf');self.min_novel=float('inf')
        self.contact_seen=False;self.collision_seen=False;self.telemetry_available=False
        mapping={sim_name(link,ident):(link,ident) for link,ident in base.SENSORS}
        self.keys=[mapping[name] for name in NAMES]
        with np.load(args.guard_baseline,allow_pickle=False) as f:
            if list(f['sensor_names'])!=NAMES:raise ValueError('guard background has a different sensor order')
            self.background=f['minimum_depth_m'].copy()
        if self.background.shape!=(40,8,8) or not np.isfinite(self.background).all():
            raise ValueError('invalid background proximity profile')

    def setup(self,cleanup,diagnostic=False):
        if self.listener is None:
            sockets=base.open_sensor_sockets()
            for sock in sockets:cleanup.callback(sock.close)
            self.listener=base.SensorListener(sockets)
            self.listener.start()
            cleanup.callback(self.listener.join,1.0)
            cleanup.callback(self.listener.stop_event.set)
        base.wait_for_sensors(self.listener,self.args.sensor_warmup,diagnostic)
        if not diagnostic:self.control_guard()

    def _abort(self,reason,snapshot=None,**details):
        if self.trigger is None:
            self.trigger=dict(reason=reason,monotonic_time=time.monotonic(),**details)
            self.events.append(self.trigger)
        raise ProximityAbort(json.dumps(self.trigger))

    def read(self,diagnostic=False):
        with self.lock:return self._read(diagnostic)

    def _read(self,diagnostic=False):
        if self.trigger is not None and not diagnostic:raise ProximityAbort(json.dumps(self.trigger))
        packets=self.listener.snapshot();now=time.monotonic()
        depth=np.full((40,128),4000,np.uint16);ages=np.full(40,np.nan,np.float32);counts=np.zeros(40,np.int64)
        faults=[None]*40
        for i,key in enumerate(self.keys):
            if key not in packets:
                if not diagnostic:self._abort('missing_sensor',sensor=NAMES[i])
                faults[i]='missing';continue
            raw,arrived,count=packets[key]
            ages[i]=(now-arrived)*1000;counts[i]=count
            if np.shape(raw)!=(128,):
                if not diagnostic:self._abort('invalid_sensor_packet',sensor=NAMES[i])
                faults[i]='bad_packet';continue
            depth[i]=raw
            if ages[i]<0 or ages[i]>self.args.sensor_timeout*1000:
                if not diagnostic:self._abort('stale_sensor',sensor=NAMES[i],age_ms=float(ages[i]))
                faults[i]='stale'
        meters=depth[:,:64].reshape(40,8,8).astype(np.float32)/1000
        meters[(meters<=0)|(meters>=4)]=4.
        usable=np.asarray([fault is None for fault in faults])
        valid=(meters>=.005)&(meters<4)&usable[:,None,None]
        novel=valid&(meters<self.background-self.args.guard_drop_m)
        close=novel&(meters<=self.args.stop_distance_m)
        snapshot=dict(depth_mm=depth.reshape(40,2,8,8),depth_m=meters,age_ms=ages,count=counts,
                      time=now,novel=novel,close=close,faults=faults)
        self.last_snapshot=snapshot
        if valid.any():self.min_raw=min(self.min_raw,float(meters[valid].min()))
        if novel.any():self.min_novel=min(self.min_novel,float(meters[novel].min()))
        sensor_counts=close.sum(axis=(1,2))
        triggered=np.flatnonzero(sensor_counts>=self.args.guard_min_zones)
        snapshot['guard_sensors']=[NAMES[i] for i in triggered]
        if len(triggered) and not diagnostic:
            i=int(triggered[np.argmax(sensor_counts[triggered])])
            self._abort('novel_proximity',snapshot,sensor=NAMES[i],zones=int(sensor_counts[i]),
                        distance_m=float(meters[i][close[i]].min()),
                        threshold_m=self.args.stop_distance_m,background_drop_m=self.args.guard_drop_m)
        return snapshot

    def control_guard(self):self.read()

    def sample(self):
        snapshot=self.read()
        snapshot['history_m']=self.history.append(snapshot['depth_m'])
        return snapshot

    def policy_input(self,snapshot):
        history=snapshot['history_m']
        return np.full_like(history,4.) if self.args.proximity_mode=='clear' else history.copy()

    def record_fields(self,snapshot):
        return dict(proximity_depth_mm=snapshot['depth_mm'],proximity_age_ms=snapshot['age_ms'],
                    proximity_packet_count=snapshot['count'],proximity_t=snapshot['time'],
                    proximity_policy_history_m=self.policy_input(snapshot),
                    proximity_novel_mask=snapshot['novel'].astype(np.uint8),**snapshot.get('robot_fields',{}))

    def robot_fields(self,robot):
        state=robot.state;fields={}
        dimensions={'joint_contact':7,'cartesian_contact':6,'joint_collision':7,'cartesian_collision':6,
                    'tau_ext_hat_filtered':7,'O_F_ext_hat_K':6}
        for name,size in dimensions.items():
            try:
                value=np.asarray(getattr(state,name),dtype=np.float64).reshape(size)
            except (AttributeError,TypeError,ValueError):value=np.full(size,np.nan)
            fields['robot_'+name]=value.copy()
        available=all(np.isfinite(fields['robot_'+name]).all() for name in list(dimensions)[:4])
        self.telemetry_available|=available
        if available:
            self.contact_seen|=bool(np.any(fields['robot_joint_contact']>0) or np.any(fields['robot_cartesian_contact']>0))
            self.collision_seen|=bool(np.any(fields['robot_joint_collision']>0) or np.any(fields['robot_cartesian_collision']>0))
        return fields

    def result_fields(self):
        return dict(policy_type='PC-ACT',proximity_mode=self.args.proximity_mode,
                    sensor_names=NAMES,proximity_guard_enabled=True,
                    guard_triggered=bool(self.trigger and self.trigger['reason']=='novel_proximity'),
                    sensor_fault=bool(self.trigger and self.trigger['reason']!='novel_proximity'),
                    stop_event=self.trigger,minimum_valid_sensor_depth_m=self.min_raw if np.isfinite(self.min_raw) else None,
                    minimum_novel_sensor_depth_m=self.min_novel if np.isfinite(self.min_novel) else None,
                    collision_free=None,learned_avoidance_demonstrated=None,
                    robot_contact_signal_seen=self.contact_seen if self.telemetry_available else None,
                    robot_collision_signal_seen=self.collision_seen if self.telemetry_available else None,
                    guard_note='Background-relative proximity abort is independent of PC-ACT. It is not learned avoidance or a certified stop.')

    def save_events(self,output,t0):
        events=[dict(e,t=e['monotonic_time']-t0) for e in self.events]
        (output/'sensor_events.json').write_text(json.dumps(events,indent=2)+'\n')
        with h5py.File(output/'rollout.h5','a') as f:
            f.attrs['sensor_names_json']=json.dumps(NAMES)
            f.attrs['proximity_mode']=self.args.proximity_mode
            if 'proximity_t' in f:
                f['proximity_t'][:]=f['proximity_t'][:]-t0
                f['proximity_t'].attrs['unit']='seconds since rollout start, host monotonic arrival snapshot'
                f['proximity_depth_mm'].attrs['layout']='tick, canonical sensor, confidence block, row, column'
                f['proximity_policy_history_m'].attrs['layout']='tick, canonical sensor, 32 causal repeated subframes, row, column; meters'
            f.create_dataset('guard/background_minimum_depth_m',data=self.background)
            if self.trigger is not None and self.last_snapshot is not None:
                f.create_dataset('guard/trigger_depth_mm',data=self.last_snapshot['depth_mm'])


def offline(args,policy,output):
    import cv2
    from contextlib import ExitStack
    records=[];queries=[];counter=[];history=CausalSkinHistory()
    player=act.ChunkPlayer(policy.query_every,args.temporal_agg,args.aggregation_decay)
    with h5py.File(args.offline_row/'trajectory.h5') as f,ExitStack() as cleanup:
        g=f['traj_0'];q=g['real/q'][:];times=g['real/t'][:];truth=g['real/target_q'][:]
        skin=np.stack([g['obs/proximity'][n][:].mean(1) for n in NAMES],axis=1).astype(np.float32)
        caps={n:cv2.VideoCapture(str(args.offline_row/f'episode_00000000_{n}.mp4')) for n in policy.camera_names}
        for cap in caps.values():cleanup.callback(cap.release)
        for tick in range(min(len(q),args.offline_steps or len(q))):
            images={}
            for name,cap in caps.items():
                ok,bgr=cap.read()
                if not ok:raise RuntimeError(f'cannot decode {name} tick {tick}')
                images[name]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            rgb=policy.prepare_images(images);live=history.append(skin[tick])
            supplied=np.full_like(live,4.) if args.proximity_mode=='clear' else live
            latency=0.
            if player.needs_query(tick):
                chunk,latency=act.predict_timed(policy,q[tick],rgb,supplied)
                player.add(tick,chunk);queries.append((tick,chunk))
                if args.compare_clear:
                    alternative=policy.predict(q[tick],rgb,np.full_like(live,4.))
                    counter.append(float(np.mean(abs(chunk[:,:7]-alternative[:,:7]))))
            records.append(dict(t=times[tick],q=q[tick],action=player.action(tick)[:7],reference_action=truth[tick],
                                inference_s=latency,rgb=rgb,proximity_policy_history_m=supplied))
    pred=np.stack([r['action'] for r in records]);target=truth[:len(records)]
    result=dict(mode='offline_pc_act_teacher_forced',success=None,checkpoint=str(args.ckpt_dir/args.ckpt_name),
        camera_names=policy.camera_names,steps=len(records),queries=len(queries),paired_epoch=policy.pair['epoch'],
        proximity_mode=args.proximity_mode,arm_mae_rad=float(abs(pred-target).mean()),
        max_arm_error_rad=float(abs(pred-target).max()),query_latency_s=[r['inference_s'] for r in records if r['inference_s']],
        live_vs_clear_mean_action_delta_rad=float(np.mean(counter)) if counter else None,
        note='Recorded observations; no hardware or obstacle-avoidance success claim.')
    act.save_results(output,records,queries,[],result)
    print(json.dumps(result,indent=2))


def configure(parser):
    parser.description=__doc__
    parser.set_defaults(ckpt_dir=DEFAULT_MODEL,no_reset=True,query_every=1,watchdog=.75)
    parser.add_argument('--reset-to-start',dest='no_reset',action='store_false',help='reset to A only with the obstacle removed and a clear path')
    parser.add_argument('--reset-only',action='store_true',help='move to A and exit, before placing the obstacle; no policy execution')
    parser.add_argument('--proximity-mode',choices=['live','clear'],default='live')
    parser.add_argument('--sensor-timeout',type=float,default=.75,help='abort if any trained sensor is older than this many seconds')
    parser.add_argument('--sensor-warmup',type=float,default=5.)
    parser.add_argument('--stop-distance-m',type=float,default=.08,help='heuristic guard threshold for novel returns, not whole-robot clearance')
    parser.add_argument('--guard-drop-m',type=float,default=.01,help='minimum decrease relative to the recorded background for a novel return')
    parser.add_argument('--guard-min-zones',type=int,default=2)
    parser.add_argument('--guard-baseline',type=Path,help='defaults to guard_baseline.npz inside checkpoint directory')
    parser.add_argument('--compare-clear',action='store_true',help='offline only: compare policy chunks with all-clear proximity')
    parser.add_argument('--sensor-check',action='store_true',help='read live UDP sensors and guard status; no robot, cameras or model')
    parser.add_argument('--sensor-check-interval',type=float,default=1.,help='seconds between full 40-sensor diagnostic tables')


def diagnostic_rows(snapshot,min_zones):
    rows=[]
    for i,name in enumerate(NAMES):
        depth=snapshot['depth_m'][i]
        valid=(depth>=.005)&(depth<4)
        fault=snapshot['faults'][i]
        close=int(snapshot['close'][i].sum())
        rows.append(dict(sensor=name,status=(fault or 'live').upper(),packets=int(snapshot['count'][i]),
            age_ms=float(snapshot['age_ms'][i]) if np.isfinite(snapshot['age_ms'][i]) else None,
            nearest_mm=float(depth[valid].min()*1000) if valid.any() and fault is None else None,
            novel_zones=int(snapshot['novel'][i].sum()),close_zones=close,
            guard_would_trigger=bool(fault is None and close>=min_zones)))
    return rows


def sensor_check(args):
    from contextlib import ExitStack
    context=LiveSkinContext(args)
    output=args.output or ROOT/'eval_rollouts'/act.datetime.now(act.timezone.utc).strftime('sensor_check_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    totals={name:dict(sensor=name,seen_packets=False,live_snapshots=0,missing_snapshots=0,
        stale_snapshots=0,bad_packet_snapshots=0,packets=0,minimum_mm=None,
        novel_return_observed=False,guard_would_trigger=False,last_status='MISSING') for name in NAMES}
    frames=0
    print('Live sensor/guard check only. No robot or camera connections will be made.',flush=True)
    print('Diagnostic mode continues after guard indications. Move the object past each sensor to check its response.',flush=True)
    try:
        with ExitStack() as cleanup:
            context.setup(cleanup,diagnostic=True)
            deadline=time.monotonic()+args.duration
            with (output/'snapshots.jsonl').open('w') as log:
                while time.monotonic()<deadline:
                    snapshot=context.read(diagnostic=True)
                    rows=diagnostic_rows(snapshot,args.guard_min_zones);frames+=1
                    live=sum(row['status']=='LIVE' for row in rows)
                    triggers=[row['sensor'] for row in rows if row['guard_would_trigger']]
                    print(f'\nSnapshot {frames}: {live}/40 LIVE; guard indications: {", ".join(triggers) or "none"}',flush=True)
                    print(f'{"SENSOR":25} {"STATUS":10} {"PACKETS":>8} {"AGE ms":>8} {"MIN mm":>8} {"NOVEL":>6} {"CLOSE":>6} {"GUARD":>6}',flush=True)
                    for row in rows:
                        age='--' if row['age_ms'] is None else f'{row["age_ms"]:.0f}'
                        near='--' if row['nearest_mm'] is None else f'{row["nearest_mm"]:.0f}'
                        print(f'{row["sensor"]:25} {row["status"]:10} {row["packets"]:8d} {age:>8} {near:>8} '
                              f'{row["novel_zones"]:6d} {row["close_zones"]:6d} {"YES" if row["guard_would_trigger"] else "-":>6}',flush=True)
                        total=totals[row['sensor']];total['last_status']=row['status']
                        total[row['status'].lower()+'_snapshots']+=1
                        total['seen_packets']|=row['packets']>0
                        total['packets']=max(total['packets'],row['packets'])
                        if row['nearest_mm'] is not None:
                            total['minimum_mm']=row['nearest_mm'] if total['minimum_mm'] is None else min(total['minimum_mm'],row['nearest_mm'])
                        total['novel_return_observed']|=row['novel_zones']>0
                        total['guard_would_trigger']|=row['guard_would_trigger']
                    log.write(json.dumps(dict(monotonic_time=snapshot['time'],sensors=rows))+'\n');log.flush()
                    remaining=deadline-time.monotonic()
                    if remaining>0:time.sleep(min(args.sensor_check_interval,remaining))
    except KeyboardInterrupt:
        print('\nSensor check stopped.',flush=True)
    finally:
        rows=list(totals.values())
        summary=dict(robot_connected=False,cameras_connected=False,snapshots=frames,sensor_count=40,
            sensors_with_packets=sum(r['seen_packets'] for r in rows),
            sensors_with_novel_returns=sum(r['novel_return_observed'] for r in rows),sensors=rows,
            note='Packets verify reception. Novel returns show background-relative changes; physical coverage and correct hardware ID mapping require checking each sensor with an object.')
        (output/'sensor_check.json').write_text(json.dumps(summary,indent=2)+'\n')
        with (output/'sensor_check.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(f'Saved all 40 sensor summaries to {output}',flush=True)


def guarded_rollout(args,policy,output,cpus,context):
    """Report expected guard stops cleanly; preserve tracebacks for unrelated bugs."""
    try:
        act.live(args,policy,output,cpus,context=context)
    except (ProximityAbort,RuntimeError) as exc:
        if context.trigger is None:raise
        if not isinstance(exc,ProximityAbort) and json.dumps(context.trigger) not in str(exc):raise
        result_path=output/'result.json'
        result=json.loads(result_path.read_text())
        preflight=result.get('termination_phase')=='sensor_preflight'
        reason=context.trigger['reason']
        result['outcome']=('blocked_at_sensor_preflight' if preflight else 'proximity_guard_stop') if reason=='novel_proximity' else 'sensor_fault'
        result['exit_code']=2
        result_path.write_text(json.dumps(result,indent=2)+'\n')
        print(f'PC-ACT {result["outcome"]}: {reason}; sensor={context.trigger.get("sensor","unknown")}.',flush=True)
        if reason=='novel_proximity':
            print(f'Measured {context.trigger["distance_m"]*1000:.1f} mm in {context.trigger["zones"]} novel close zones; '
                  f'guard threshold {context.trigger["threshold_m"]*1000:.1f} mm.',flush=True)
        if preflight:print('The check stopped before connecting to the robot or starting motion.',flush=True)
        print(f'Details saved in {result_path} and {output/"sensor_events.json"}. '
              'Use --sensor-check to inspect all sensor readings.',flush=True)
        return 2
    return 0


def main(argv=None):
    args=act.parse_args(argv,configure)
    for key in ['sensor_timeout','sensor_warmup','stop_distance_m','guard_drop_m','sensor_check_interval']:
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0:raise ValueError(f'{key} must be positive and finite')
    if not 1<=args.guard_min_zones<=64:raise ValueError('guard-min-zones must be 1..64')
    if args.compare_clear and not args.offline_row:raise ValueError('compare-clear is an offline diagnostic')
    args.guard_baseline=args.guard_baseline or args.ckpt_dir/'guard_baseline.npz'
    if args.reset_only:
        from franky import Robot,JointMotion
        robot=Robot(args.host);robot.recover_from_errors();robot.relative_dynamics_factor=args.dynamics
        robot.move(JointMotion(args.start))
        error=float(abs(base.read_robot_state(robot)['q']-args.start).max())
        if error>args.goal_tolerance:raise RuntimeError(f'reset error {error:.4f} rad exceeds start tolerance')
        print(f'Arm at A; maximum joint error {error:.4f} rad. Reset-only complete.');return
    if args.sensor_check:
        sensor_check(args);return
    output=args.output or ROOT/'eval_rollouts'/act.datetime.now(act.timezone.utc).strftime('pc_act_%Y%m%dT%H%M%S%fZ')
    if not args.check_model and output.exists():raise FileExistsError(output)
    affinity=os.sched_getaffinity(0)
    try:
        cpus=None if args.offline_row or args.check_model else base.reserve_control_cpus(args.control_cpu)
        policy=PCInference(args)
        dummy=np.zeros((len(policy.camera_names),policy.height,policy.width,3),np.uint8)
        started=time.monotonic();chunk=policy.predict(np.asarray(args.start),dummy,np.full((40,32,8,8),4.,np.float32))
        elapsed=time.monotonic()-started
        if args.check_model:
            print(json.dumps(dict(action_shape=list(chunk.shape),finite=bool(np.isfinite(chunk).all()),paired_epoch=policy.pair['epoch'],
                                  inference_s=elapsed,hardware_connected=False,device=policy.device),indent=2));return
        output.mkdir(parents=True,exist_ok=False)
        (output/'config.json').write_text(json.dumps(dict(vars(args),policy_pair=policy.pair,
            sensor_names=NAMES,resolved_camera_names=policy.camera_names,resolved_device=policy.device),default=str,indent=2)+'\n')
        if args.offline_row:offline(args,policy,output)
        else:return guarded_rollout(args,policy,output,cpus,LiveSkinContext(args))
    finally:os.sched_setaffinity(0,affinity)


if __name__=='__main__':raise SystemExit(main())
