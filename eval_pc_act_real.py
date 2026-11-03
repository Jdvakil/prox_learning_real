"""Real FR3 PC-ACT evaluation with live 40-sensor histories and independent intrusion guard.

The policy has not been trained to avoid obstacles. The guard detects novel near
returns against the recorded background; it is a heuristic abort, not learned
avoidance, a robot-clearance estimate, or a certified collision-prevention system.
"""
from collections import deque
import hashlib
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

    def setup(self,cleanup):
        if self.listener is None:
            sockets=base.open_sensor_sockets()
            for sock in sockets:cleanup.callback(sock.close)
            self.listener=base.SensorListener(sockets)
            self.listener.start()
            cleanup.callback(self.listener.join,1.0)
            cleanup.callback(self.listener.stop_event.set)
        base.wait_for_sensors(self.listener,self.args.sensor_warmup,False)
        self.control_guard()

    def _abort(self,reason,snapshot=None,**details):
        if self.trigger is None:
            self.trigger=dict(reason=reason,monotonic_time=time.monotonic(),**details)
            self.events.append(self.trigger)
        raise ProximityAbort(json.dumps(self.trigger))

    def read(self):
        with self.lock:return self._read()

    def _read(self):
        if self.trigger is not None:raise ProximityAbort(json.dumps(self.trigger))
        packets=self.listener.snapshot();now=time.monotonic()
        depth=np.empty((40,128),np.uint16);ages=np.empty(40,np.float32);counts=np.empty(40,np.int64)
        for i,key in enumerate(self.keys):
            if key not in packets:self._abort('missing_sensor',sensor=NAMES[i])
            raw,arrived,count=packets[key]
            if np.shape(raw)!=(128,):self._abort('invalid_sensor_packet',sensor=NAMES[i])
            depth[i]=raw;ages[i]=(now-arrived)*1000;counts[i]=count
            if ages[i]<0 or ages[i]>self.args.sensor_timeout*1000:
                self._abort('stale_sensor',sensor=NAMES[i],age_ms=float(ages[i]))
        meters=depth[:,:64].reshape(40,8,8).astype(np.float32)/1000
        meters[(meters<=0)|(meters>=4)]=4.
        valid=(meters>=.005)&(meters<4)
        novel=valid&(meters<self.background-self.args.guard_drop_m)
        close=novel&(meters<=self.args.stop_distance_m)
        snapshot=dict(depth_mm=depth.reshape(40,2,8,8),depth_m=meters,age_ms=ages,count=counts,
                      time=now,novel=novel,close=close)
        self.last_snapshot=snapshot
        if valid.any():self.min_raw=min(self.min_raw,float(meters[valid].min()))
        if novel.any():self.min_novel=min(self.min_novel,float(meters[novel].min()))
        sensor_counts=close.sum(axis=(1,2))
        triggered=np.flatnonzero(sensor_counts>=self.args.guard_min_zones)
        if len(triggered):
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


def sensor_check(args):
    from contextlib import ExitStack
    context=LiveSkinContext(args)
    print('Live sensor/guard check only. No robot or camera connections will be made.',flush=True)
    try:
        with ExitStack() as cleanup:
            context.setup(cleanup)
            deadline=time.monotonic()+args.duration
            while time.monotonic()<deadline:
                snapshot=context.read();novel=snapshot['novel'];depth=snapshot['depth_m']
                print(json.dumps(dict(max_age_ms=float(snapshot['age_ms'].max()),
                    nearest_valid_m=float(depth[(depth>=.005)&(depth<4)].min()) if ((depth>=.005)&(depth<4)).any() else None,
                    nearest_novel_m=float(depth[novel].min()) if novel.any() else None,
                    novel_zones=int(novel.sum()),guard_triggered=False)),flush=True)
                time.sleep(.2)
    except ProximityAbort:
        print(json.dumps(dict(robot_connected=False,cameras_connected=False,guard_would_abort=context.trigger),indent=2))


def main(argv=None):
    args=act.parse_args(argv,configure)
    for key in ['sensor_timeout','sensor_warmup','stop_distance_m','guard_drop_m']:
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
        else:act.live(args,policy,output,cpus,context=LiveSkinContext(args))
    finally:os.sched_setaffinity(0,affinity)


if __name__=='__main__':main()
