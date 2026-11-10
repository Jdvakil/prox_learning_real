"""Read-only full audit of a recorded A -> B batch (prox_real.analysis.run sets SOURCE and OUT)."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from ..record_act_waypoints import ReferencePlan
from ..kinesthetic import Q_MIN, Q_MAX

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'act_20_analysis'

SOURCE = ROOT / 'recordings/act_test_20'


def stats(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if not a.size:
        return {}
    return dict(zip(('min', 'p01', 'p05', 'median', 'p95', 'p99', 'max'),
                    map(float, np.percentile(a, [0, 1, 5, 50, 95, 99, 100]))))


def decode(ds):
    return [json.loads(bytes(v).rstrip(b'\0')) for v in ds[:]]


def video(path, preview_indices):
    cap = cv2.VideoCapture(str(path))
    meta = dict(fps=float(cap.get(cv2.CAP_PROP_FPS)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    small, bright, dark, clipped, sharp, changes, previews = [], [], [], [], [], [], []
    previous = None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        i = len(small)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(im, (160, 120), interpolation=cv2.INTER_AREA)
        small.append(thumb)
        bright.append(float(gray.mean()))
        dark.append(float(np.mean(gray <= 2)))
        clipped.append(float(np.mean(gray >= 253)))
        sharp.append(float(cv2.Laplacian(cv2.resize(gray, (320, 240)), cv2.CV_64F).var()))
        if previous is not None:
            changes.append(float(np.mean(np.abs(thumb.astype(float) - previous))))
        previous = thumb.astype(float)
        if i in preview_indices:
            previews.append((i, cv2.resize(im, (320, 240))))
    cap.release()
    meta.update(decoded_frames=len(small), brightness=stats(bright), dark_pixel_fraction=stats(dark),
                saturated_pixel_fraction=stats(clipped), sharpness_laplacian_variance=stats(sharp),
                adjacent_thumbnail_mae=stats(changes), exact_repeated_adjacent_thumbnails=int(np.sum(np.asarray(changes) == 0)),
                nearly_black_frames=int(np.sum(np.asarray(bright) < 2)),
                nearly_white_frames=int(np.sum(np.asarray(bright) > 253)))
    return meta, np.asarray(small), previews


def pca(x, standardized=False):
    mean = x.mean(axis=0)
    scale = x.std(axis=0) if standardized else np.ones(x.shape[1])
    scale = np.where(scale > 1e-12, scale, 1)
    z = (x - mean) / scale
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    variance = s**2 / (len(x) - 1)
    # Deterministic signs for reproducible plots; PCA directions are sign-indeterminate.
    for row in vt:
        if row[np.argmax(abs(row))] < 0:
            row *= -1
    ratio = variance / variance.sum()
    return dict(mean=mean, scale=scale, components=vt, scores=z @ vt.T,
                explained_variance=variance, explained_variance_ratio=ratio,
                cumulative_explained_variance=np.cumsum(ratio))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = json.loads((SOURCE / 'config.json').read_text())
    rows = sorted(SOURCE.glob('episode_*/trajectory.h5'))
    episodes, cameras, arrays, issues, warnings = [], [], {}, [], []
    q_all, v_all, ref_all, t_all, clock_all, ids, phase_q, phase_v, residuals, noise_phase = [], [], [], [], [], [], [], [], [], []
    phase = np.linspace(0, config['duration'], 121)
    sensor_age, sensor_fresh, camera_series = [], [], []
    sample_episodes = {0, len(rows)//2, len(rows)-1}
    for ep, path in enumerate(rows):
        result = json.loads((path.parent / 'result.json').read_text())
        with h5py.File(path) as f:
            g = f['traj_0']; r = g['real']
            t, q, v, ref, clock = (r[k][:] for k in ('t', 'q', 'dq', 'target_q', 'clock'))
            ct, cq, cv = (r['control'][k][:] for k in ('t', 'q_ref', 'dq_ref'))
            seed = int(g.attrs['seed'])
            plan = ReferencePlan(g.attrs['start_q'], g.attrs['goal_q'],
                                 float(g.attrs['duration_s']), float(g.attrs['noise_std_rad']), seed)
            expected, expected_v, _, _ = plan.evaluate(clock)
            idx = np.searchsorted(ct, t, side='right') - 1
            live = idx >= 0
            action_json = decode(g['actions/joint_pos'])
            action = np.asarray([x['arm'] for x in action_json])
            pos_json = np.asarray([x['arm'] for x in decode(g['obs/agent/qpos'])])
            vel_json = np.asarray([x['arm'] for x in decode(g['obs/agent/qvel'])])
            duplicate_action = np.asarray([x['arm'] for x in decode(g['actions/commanded_action'])])
            coeff = r['noise_coefficients'][:]
            plan_file = np.load(path.parent / 'reference_plan.npz')
            finite = all(np.isfinite(x).all() for x in [t,q,v,ref,ct,cq,cv,clock])
            moving = (clock > 0) & (clock < config['duration'])
            selected = np.flatnonzero(moving)
            # Include the last lead-in sample and first goal-hold sample to bracket [0,T].
            sel = np.arange(max(0,selected[0]-1), min(len(t),selected[-1]+2))
            pq = np.stack([np.interp(phase, clock[sel], q[sel,j]) for j in range(7)],axis=1)
            pv = np.stack([np.interp(phase, clock[sel], v[sel,j]) for j in range(7)],axis=1)
            nominal = plan.evaluate(phase)[2]
            phase_q.append(pq); phase_v.append(pv); residuals.append(pq-nominal)
            noise_phase.append(plan.evaluate(phase)[3])
            err = q-ref
            dqd = np.gradient(q, t, axis=0)
            item = dict(episode=path.parent.name, seed=seed, ticks=len(t), duration_s=float(t[-1]-t[0]),
                        clean_success=bool(result.get('clean_success')), tick_dt_ms=stats(np.diff(t)*1000),
                        control_dt_ms=stats(np.diff(ct)*1000), control_samples=len(ct),
                        max_tracking_error_rad=float(abs(err).max()), tracking_rmse_rad=float(np.sqrt(np.mean(err**2))),
                        final_goal_error_rad=float(abs(q[-1]-plan.goal).max()),
                        start_error_rad=float(abs(q[0]-plan.start).max()),
                        final_max_speed_rad_s=float(abs(v[-1]).max()),
                        max_speed_rad_s=float(abs(v).max()), peak_reference_speed_rad_s=float(abs(cv).max()),
                        max_noise_rad=float(abs(r['noise_q'][:]).max()),
                        action_vs_reference_error=float(abs(action-ref).max()),
                        duplicate_action_error=float(abs(duplicate_action-ref).max()),
                        qpos_vs_q_error=float(abs(pos_json-q).max()), qvel_vs_dq_error=float(abs(vel_json-v).max()),
                        action_vs_active_command_error=float(abs(ref[live]-cq[idx[live]]).max()),
                        action_vs_seeded_plan_error=float(abs(ref-expected).max()),
                        velocity_reference_error=float(abs(r['dq_ref'][:]-expected_v).max()),
                        saved_coefficients_match_seed=bool(np.array_equal(coeff,plan.coefficients)),
                        saved_plan_matches_coefficients=bool(np.array_equal(coeff,plan_file['coefficients'])),
                        last_tick_minus_last_command_ms=float((t[-1]-ct[-1])*1000),
                        velocity_vs_finite_difference_rmse_rad_s=float(np.sqrt(np.mean((dqd-v)**2))),
                        finite=bool(finite), monotonic_times=bool(np.all(np.diff(t)>0) and np.all(np.diff(ct)>0)),
                        monotonic_clock=bool(np.all(np.diff(clock)>=0)), joint_limits_valid=bool(np.all(q>=Q_MIN) and np.all(q<=Q_MAX)),
                        nonarm_action_keys=sorted(set().union(*(set(x)-{'arm','base'} for x in action_json))),
                        quaternion_norm_error=float(abs(np.linalg.norm(g['obs/extra/tcp_pose'][:,3:],axis=1)-1).max()),
                        success_indices=np.flatnonzero(g['success'][:]).tolist(),
                        fail_indices=np.flatnonzero(g['fail'][:]).tolist(),
                        termination_indices=np.flatnonzero(g['terminated'][:]).tolist())
            failures=[]
            for key in ('finite','monotonic_times','monotonic_clock','joint_limits_valid','saved_coefficients_match_seed','saved_plan_matches_coefficients','clean_success'):
                if not item[key]: failures.append(key)
            for key in ('action_vs_reference_error','duplicate_action_error','qpos_vs_q_error','qvel_vs_dq_error','action_vs_active_command_error','action_vs_seeded_plan_error','velocity_reference_error'):
                if item[key]>1e-8:failures.append(key)
            if item['final_goal_error_rad']>config['goal_tolerance']:failures.append('goal_tolerance')
            if item['max_tracking_error_rad']>config['max_error']:failures.append('tracking_limit')
            if t[-1]>ct[-1]+0.02:failures.append('post_control_observations')
            if item['fail_indices'] or item['success_indices'] != [len(t)-1] or item['termination_indices'] != [len(t)-1]:failures.append('outcome_flags')
            item['failures']=failures
            issues.extend(f'{path.parent.name}: {x}' for x in failures)
            ages=r['age_ms'][:];fresh=r['fresh'][:]
            item['sensor_count']=ages.shape[1];item['sensor_age_ms']=stats(ages)
            item['sensor_nonfinite_ages']=int(np.sum(~np.isfinite(ages)));item['sensor_negative_ages']=int(np.sum(ages<0))
            sensor_age.append(ages);sensor_fresh.append(fresh)
            prox=r['depth_mm'][:];item['proximity_valid_fraction']=float(np.mean((prox>0)&(prox<4000)))
            item['sensor_names']=r['sensor_names'].asstr()[:].tolist()
            timestamps=[]
            for name,c in r['cameras'].items():
                ft,nums,ti=(c[k][:] for k in ('t','frame_number','tick_frame'))
                invalid_index=int(np.sum((ti<0)|(ti>=len(ft))))
                safe=np.clip(ti,0,len(ft)-1)
                cm=dict(episode=path.parent.name,camera=name,frames=len(ft),serial=str(c.attrs['serial']),
                        width=int(c.attrs['width']),height=int(c.attrs['height']),fps=float(c.attrs['fps']),
                        fx=float(c.attrs['fx']),fy=float(c.attrs['fy']),depth_scale_m=float(c.attrs['depth_scale_m']),
                        depth_dtype=str(c['depth_mm'].dtype),depth_shape=list(c['depth_mm'].shape),
                        first_frame_s=float(ft[0]),last_frame_s=float(ft[-1]),frame_dt_ms=stats(np.diff(ft)*1000),
                        dropped_frame_numbers=int(np.maximum(np.diff(nums)-1,0).sum()),
                        nonincreasing_frame_numbers=int(np.sum(np.diff(nums)<=0)),
                        nonincreasing_times=int(np.sum(np.diff(ft)<=0)),
                        nonincreasing_device_times=int(np.sum(np.diff(c['device_time_ms'][:])<=0)),
                        invalid_tick_indices=invalid_index,future_tick_frames=int(np.sum(ft[safe]>t+0.001)),
                        tick_age_ms=stats((t-ft[safe])*1000))
                timestamps.append(ft[safe]);validity=[];near=[];far=[];sat=[];depth_changes=[];prev=None
                hist=np.zeros(65536,dtype=np.int64)
                for start in range(0,len(ft),16):
                    d=c['depth_mm'][start:start+16]
                    hist+=np.bincount(d.ravel(),minlength=65536)
                    validity.extend(np.mean(d>0,axis=(1,2)).tolist())
                    near.extend(np.mean((d>0)&(d<100),axis=(1,2)).tolist())
                    far.extend(np.mean(d>10000,axis=(1,2)).tolist())
                    sat.extend(np.mean(d==65535,axis=(1,2)).tolist())
                    for im in d[:,::8,::8]:
                        if prev is not None:
                            mask=(im>0)&(prev>0)
                            depth_changes.append(float(np.mean(abs(im[mask].astype(float)-prev[mask]))) if mask.any() else 0)
                        prev=im.copy()
                cm.update(depth_valid_fraction=stats(validity),depth_lt_100mm_fraction=stats(near),
                          depth_gt_10m_fraction=stats(far),depth_saturated_fraction=stats(sat),
                          empty_depth_frames=int(np.sum(np.asarray(validity)==0)),
                          adjacent_depth_change_mm=stats(depth_changes),mean_depth_valid_fraction=float(np.mean(validity)))
                quantile_indices=np.searchsorted(np.cumsum(hist[1:]),np.array([.01,.5,.99])*hist[1:].sum())+1
                cm['depth_valid_mm_p01_median_p99']=quantile_indices.tolist()
                arrays[f'depth_hist_{ep}_{name}']=hist
                sampled=np.linspace(0,len(t)-1,5,dtype=int)
                tickmeta,ticksmall,previews=video(path.parent/f'episode_00000000_{name}.mp4',sampled)
                fullmeta,fullsmall,_=video(path.parent/f'{name}_30fps.mp4',[])
                cm['tick_video']=tickmeta;cm['full_video']=fullmeta
                if len(fullsmall)==len(ft) and len(ticksmall)==len(t):
                    pair_mae=np.mean(abs(ticksmall.astype(float)-fullsmall[safe].astype(float)),axis=(1,2,3))
                    cm['rgb_tick_vs_indexed_full_mae_0_255']=stats(pair_mae)
                errors=[]
                if tickmeta['decoded_frames']!=len(t):errors.append('RGB tick count mismatch')
                if fullmeta['decoded_frames']!=len(ft):errors.append('RGB/depth full count mismatch')
                for vm in (tickmeta,fullmeta):
                    if vm['width']!=640 or vm['height']!=480:errors.append('RGB resolution mismatch')
                    if vm['nearly_black_frames'] or vm['nearly_white_frames']:errors.append('blank RGB frames')
                if tickmeta['fps']!=15 or fullmeta['fps']!=30:errors.append('video fps mismatch')
                if cm['invalid_tick_indices']:errors.append('invalid tick/depth association')
                if cm['tick_age_ms']['min'] < -10:errors.append('camera precedes tick timestamp by >10ms')
                elif cm['future_tick_frames']:
                    warnings.append(f"{path.parent.name}/{name}: {cm['future_tick_frames']} frames arrive 1–10ms "
                                    "after the tick timestamp; latest camera is read after that timestamp")
                if cm['nonincreasing_times'] or cm['nonincreasing_device_times'] or cm['nonincreasing_frame_numbers']:errors.append('non-monotonic camera stream')
                if cm['empty_depth_frames']:errors.append('empty depth frames')
                if cm['tick_age_ms']['max']>100:errors.append('camera age >100ms')
                if cm['depth_dtype']!='uint16' or cm['depth_shape'][1:]!=[480,640]:errors.append('depth format')
                cm['failures']=errors;issues.extend(f'{path.parent.name}/{name}: {x}' for x in errors)
                cameras.append(cm)
                camera_series.append(dict(episode=ep,camera=name,frame_t=ft.tolist(),valid_fraction=validity))
                if ep in sample_episodes:
                    rgbtiles=[];depthtiles=[]
                    for n,im in previews:
                        cv2.putText(im,f'{path.parent.name} {name} t={t[n]:.1f}s',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,255,0),1)
                        rgbtiles.append(im)
                        d=c['depth_mm'][int(safe[n])];im=cv2.applyColorMap(np.uint8(np.clip(d/2000,0,1)*255),cv2.COLORMAP_TURBO);im[d==0]=0
                        im=cv2.resize(im,(320,240));cv2.putText(im,f'depth 0-2m; invalid black t={t[n]:.1f}s',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1);depthtiles.append(im)
                    cv2.imwrite(str(OUT/f'preview_ep{ep:02d}_{name}.jpg'),np.vstack([np.hstack(rgbtiles),np.hstack(depthtiles)]))
            item['intercamera_tick_skew_ms']=stats(abs(timestamps[0]-timestamps[1])*1000)
            episodes.append(item)
            q_all.append(q);v_all.append(v);ref_all.append(ref);t_all.append(t);clock_all.append(clock);ids.extend([ep]*len(t))
            print(f"{path.parent.name}: {len(t)} ticks, tracking {item['max_tracking_error_rad']:.5f} rad, camera checks complete",flush=True)
    q,v,ref,t,clock,ids=map(np.concatenate,(q_all,v_all,ref_all,t_all,clock_all,[np.asarray(ids)]))
    phase_q=np.asarray(phase_q);phase_v=np.asarray(phase_v);residuals=np.asarray(residuals);noise_phase=np.asarray(noise_phase)
    arrays.update(q=q,v=v,ref=ref,t=t,clock=clock,episode_id=ids,phase=phase,phase_q=phase_q,phase_v=phase_v,
                  residual=residuals,noise_phase=noise_phase,sensor_age=np.concatenate(sensor_age),sensor_fresh=np.concatenate(sensor_fresh))
    pca_summary={}
    for name,x,scaled in [('position',q,False),('velocity',v,False),('combined',np.c_[q,v],True),
                          ('motion_residual',residuals.reshape(-1,7),False)]:
        pc=pca(x,scaled)
        for key,value in pc.items():arrays[f'pca_{name}_{key}']=value
        ratio=pc['explained_variance_ratio'];cum=pc['cumulative_explained_variance']
        pca_summary[name]=dict(standardized=scaled,explained_variance_ratio=ratio.tolist(),
                              cumulative_variance=cum.tolist(),components_for_95pct=int(np.searchsorted(cum,.95)+1),
                              components_for_99pct=int(np.searchsorted(cum,.99)+1))
        features=[f'{"dq" if name=="velocity" else "q"}{j+1}' for j in range(7)]+([f'dq{j+1}' for j in range(7)] if scaled else [])
        write_csv(OUT/f'pca_{name}_components.csv',[dict(component=i+1,explained_variance_ratio=float(ratio[i]),
                  **dict(zip(features,map(float,pc['components'][i])))) for i in range(len(ratio))])
    coefficients=[hashlib.sha256(np.asarray(ReferencePlan(config['start'],config['goal'],config['duration'],config['noise_std'],e['seed']).coefficients).tobytes()).hexdigest() for e in episodes]
    joint_stats=[]
    for j in range(7):
        joint_stats.append(dict(joint=j+1,q_mean_rad=float(q[:,j].mean()),q_std_rad=float(q[:,j].std()),
                                q_min_rad=float(q[:,j].min()),q_max_rad=float(q[:,j].max()),
                                dq_mean_rad_s=float(v[:,j].mean()),dq_std_rad_s=float(v[:,j].std()),
                                dq_min_rad_s=float(v[:,j].min()),dq_max_rad_s=float(v[:,j].max()),
                                tracking_rmse_rad=float(np.sqrt(np.mean((q[:,j]-ref[:,j])**2))),
                                phase_aligned_q_std_mean_rad=float(phase_q[:,:,j].std(axis=0).mean()),
                                phase_aligned_dq_std_mean_rad_s=float(phase_v[:,:,j].std(axis=0).mean())))
    summary=dict(source=str(SOURCE),episodes=len(episodes),expected_episodes=config['episodes'],ticks=len(q),
                 total_duration_s=float(sum(e['duration_s'] for e in episodes)),
                 unique_seeds=len(set(e['seed'] for e in episodes)),unique_noise_coefficients=len(set(coefficients)),
                 failures=issues,warnings=warnings,episode_metrics=episodes,camera_metrics=cameras,joint_statistics=joint_stats,pca=pca_summary,
                 sensor_age_ms=stats(arrays['sensor_age']),
                 method_notes=['All HDF5 depth frames read; all four videos per episode fully decoded.',
                               'RGB content comparisons use 160x120 thumbnails after decoding; codec loss prevents exact RGB equality.',
                               'Depth zero denotes invalid; >10m is reported rather than automatically deemed invalid.',
                               'Position/velocity PCA are separately mean-centered without scaling, in rad and rad/s respectively.',
                               'Combined 14D PCA standardizes each feature to unit population SD before SVD.',
                               'Residual PCA uses phase-aligned measured q minus analytical nominal A-to-B path on 121 points per episode.',
                               'Depth correctness in meters and RGB-depth extrinsic accuracy require a known physical target; this audit checks structure, range, temporal consistency, and previews.',
                               'Samples in an episode are temporally correlated; statistical spreads are descriptive, not independent-sample confidence intervals.'])
    if len(episodes)!=config['episodes']:summary['failures'].append('episode count mismatch')
    if summary['unique_seeds']!=len(episodes):summary['failures'].append('duplicate seeds')
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
    (OUT/'camera_series.json').write_text(json.dumps(camera_series))
    np.savez_compressed(OUT/'analysis_arrays.npz',**arrays)
    write_csv(OUT/'joint_statistics.csv',joint_stats)
    write_csv(OUT/'episode_statistics.csv',[{k:e[k] for k in ['episode','seed','ticks','duration_s','max_tracking_error_rad','tracking_rmse_rad','final_goal_error_rad','final_max_speed_rad_s','max_noise_rad','max_speed_rad_s','last_tick_minus_last_command_ms','velocity_vs_finite_difference_rmse_rad_s']} for e in episodes])
    print('Saved audit metrics:',OUT/'metrics.json',flush=True)
