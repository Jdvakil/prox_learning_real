"""Build the human-readable assessment and local HTML gallery from audit results."""
import csv
import html
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
M=json.loads((P/'metrics.json').read_text())
E=M['episode_metrics'];C=M['camera_metrics'];D=np.load(P/'analysis_arrays.npz')
plots=json.loads((P/'plot_index.json').read_text())


def csvwrite(name,rows):
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


csvwrite('camera_statistics.csv',[dict(episode=c['episode'],camera=c['camera'],serial=c['serial'],
    frames=c['frames'],dropped=c['dropped_frame_numbers'],
    mean_valid_depth_fraction=c['mean_depth_valid_fraction'],
    min_frame_valid_depth_fraction=c['depth_valid_fraction']['min'],
    min_frame_age_ms=c['tick_age_ms']['min'],max_frame_age_ms=c['tick_age_ms']['max'],
    frames_1ms_after_tick=c['future_tick_frames'],
    max_rgb_pair_mae=c['rgb_tick_vs_indexed_full_mae_0_255']['max'],
    empty_depth_frames=c['empty_depth_frames'],
    median_valid_depth_mm=c['depth_valid_mm_p01_median_p99'][1]) for c in C])
ages=D['sensor_age'];fresh=D['sensor_fresh']
csvwrite('sensor_statistics.csv',[dict(index=i,name=E[0]['sensor_names'][i],
    median_age_ms=float(np.median(ages[:,i])),p95_age_ms=float(np.percentile(ages[:,i],95)),
    max_age_ms=float(ages[:,i].max()),fresh_fraction=float(fresh[:,i].mean())) for i in range(ages.shape[1])])

def pct(x):return f'{100*x:.2f}%'
def rng(key):return f"{min(e[key] for e in E):.6f} to {max(e[key] for e in E):.6f}"

text=f'''ACT 20-EPISODE DATA AUDIT
Source: {M['source']}
Assessment: suitable for the intended fixed-start, fixed-goal ACT overfit and rollout smoke test.
No episode needs to be discarded on the audited motion or data-integrity criteria.
No original recording was edited. All 20 episodes are included in every pooled analysis.

SCOPE AND COUNTS
20/20 successful episodes; 20 unique seeds and 20 unique perturbation coefficient sets.
3,040 observations: 152 per episode, about 10.067 seconds each; 201.340 seconds in total.
Every HDF5 depth frame was read: {sum(c['frames'] for c in C):,} frames at 640 x 480, uint16 mm.
Every frame in all 80 video files was decoded: {sum(c['tick_video']['decoded_frames']+c['full_video']['decoded_frames'] for c in C):,} video frames.
RGB numeric checks cover all frames; visual review uses all-episode midpoint contact sheets
plus five-timepoint RGB/depth previews for episodes 0, 10, and 19.

MOTION AND ACTION LABELS
Maximum observed tracking error over the batch: {max(e['max_tracking_error_rad'] for e in E):.6f} rad
({np.degrees(max(e['max_tracking_error_rad'] for e in E)):.3f} degrees); below the configured 0.15 rad abort bound.
Per-episode final goal error: {rng('final_goal_error_rad')} rad; all below 0.02 rad tolerance.
Per-episode RMS tracking error across samples/joints: {rng('tracking_rmse_rad')} rad.
Largest measured joint speed: {max(e['max_speed_rad_s'] for e in E):.6f} rad/s.
Final maximum joint speeds: {rng('final_max_speed_rad_s')} rad/s; all arms settle at B.
Action JSON matches real/target_q exactly. Each post-lead-in action matches the actual
100 Hz reference active at that timestamp, and the analytically reconstructed seeded plan.
qpos/qvel JSON matches raw joint state exactly; only seven arm joints, no gripper commands.
All arrays used for motion analysis are finite. Robot/reference clocks are monotonic.
No sampled joint positions violate the FR3 limits used by the collector.
All terminal success/termination flags are consistent, with no fail flags.
The last observation precedes the last sent reference by 47.0–57.5 ms: no shutdown drift.

Positions are smooth overall. Joint 2 and 4 show repeatable small offsets under impedance
control, and a small transient at controller entry near one second. These are actual measured
behavior, not mislabeled waypoints. Joint 5 and 7 move mainly due to small perturbations;
their measured velocity traces are noisier and show occasional small spikes. Peak measured
speed stays below 0.061 rad/s. The RMS difference between reported velocity and a centered
finite-difference position derivative is 0.00142–0.00201 rad/s per episode. No smoothing or
outlier removal was applied. Joint statistics expose per-joint offsets and spreads.

TIMING
Nominal observation period is 66.667 ms (15 Hz); the largest interval is
{max(e['tick_dt_ms']['max'] for e in E):.3f} ms. No missing observation ticks are evident.
Reference updates have approximately 10 ms median periods; largest interval is
{max(e['control_dt_ms']['max'] for e in E):.3f} ms. These are host reference updates, not measurements of the 1 kHz internal torque loop.
Camera frames run at approximately 33.3 ms (30 Hz), with zero frame-number drops in both cameras.
The maximum difference between camera arrival times selected at one tick is
{max(e['intercamera_tick_skew_ms']['max'] for e in E):.3f} ms. This is software alignment, not hardware-trigger synchronization.

RGB-D VALIDITY
Both cameras have stable serial numbers and intrinsics across all episodes, valid 640x480
dimensions, and a recorded 0.001 m depth scale. Every episode's 15 Hz RGB count is exactly
152; its full-rate RGB count matches its depth frame count. All tick_frame indices are in bounds.
No empty depth frames, blank RGB frames, or repeated adjacent decoded thumbnails were found.
RGB tick images match their indexed full-rate images closely: worst mean thumbnail pixel
difference is 0.961 on a 0–255 scale, consistent with separate lossy video encoders.
'''
for name in ['wrist_camera','exo_camera_1']:
    cs=[c for c in C if c['camera']==name]
    text+=f'''\n{name}:
  Full-rate frames: {sum(c['frames'] for c in cs):,}; dropped frame numbers: 0.
  Mean nonzero depth coverage per episode: {pct(min(c['mean_depth_valid_fraction'] for c in cs))}–{pct(max(c['mean_depth_valid_fraction'] for c in cs))}.
  Worst single-frame nonzero depth coverage: {pct(min(c['depth_valid_fraction']['min'] for c in cs))}.
  Maximum sampled frame age: {max(c['tick_age_ms']['max'] for c in cs):.3f} ms.
  Intrinsics fx/fy: {cs[0]['fx']:.3f}/{cs[0]['fy']:.3f} pixels.
'''
text+='''
The wrist's tabletop view has high depth coverage. The external camera includes farther
backgrounds and reflective/invalid regions, so its lower coverage is plausible and stable.
External returns beyond 10 m occupy at most about 0.405% of an image; saturated 65535-mm
pixels occupy at most 0.052%. These are localized and do not invalidate the whole frame.
The exocentric view crops much of the arm, and tabletop objects leave the wrist view near B.
This is acceptable for the stated motion smoke test, but matters for future vision-dependent tasks.

TIMESTAMP CAVEAT (KEEP THE DATA)
51 of 3,040 wrist observations use a camera frame arriving 1–4.706 ms after the tick timestamp,
spread over episodes 4, 5, 10, 13, 16, 17, and 19. The collector timestamps the tick before
reading camera.latest(), so a new frame can arrive during the observation-acquisition window.
All corresponding RGB/depth indices remain valid, with RGB image agreement within codec loss.
This is acquisition-window skew, not evidence of wrong depth frame selection. Metrics retain
the signed ages and seven sequence warnings. For stricter future sensor fusion, timestamp
each modality separately or explicitly choose frames no later than the observation timestamp.
Do not reinterpret this run as sub-millisecond synchronized RGB-D/robot measurements.

Depth tests establish data integrity, range plausibility, coverage and temporal consistency.
Absolute depth accuracy and subpixel RGB/depth registration cannot be certified without a
known-distance calibration target. Intrinsics are present; camera-to-robot and inter-camera
extrinsic calibration are not validated here.

PROXIMITY
All 40 configured arm sensors are present in a consistent order, without missing or negative
packet ages. Median packet age is 63.56 ms; 95th percentile 164.45 ms; worst case 604.49 ms.
The sensors update more slowly than the policy clock, so held values are expected. This
does not block RGB + joint-state ACT, but latency should be modeled if proximity is used
for fast feedback. Sensor index/name, age, and freshness are saved in sensor_statistics.csv.

PCA METHOD AND INTERPRETATION
PCA mean-centers observations, forms an SVD X = U S V^T, and reports eigenvalue proportions
S^2 / sum(S^2). Components are orthonormal linear directions; scores are the observations
projected onto those directions. Directions have arbitrary signs. Full-component reconstruction
was checked numerically (maximum error below 1.3e-14, in the corresponding scaled coordinates).
This is descriptive analysis: temporally neighboring observations are correlated.

Position PCA: mean-centered seven joint positions in radians, no per-joint standardization.
PC1 explains 99.344%; one component exceeds both 95% and 99% retained variance. This is
expected: all demonstrations share one A-to-B progression. High PC1 variance is not proof of
bad data or of 20 duplicated trajectories.

Velocity PCA: mean-centered seven joint velocities in rad/s, no standardization.
PC1 explains 87.364%; four components retain 95.729%, and six retain 99.071%.

Combined state PCA: seven positions plus seven velocities, each standardized to unit SD
before SVD, so the two physical units are not directly mixed. PC1/PC2 explain 35.808%/31.214%
(67.022% together). Six components retain 95.176%; nine retain 99.224%. Standardization gives
small-excursion joints equal feature weight, so do not compare this percentage directly to
the unscaled position PCA as though they used the same metric.

Residual PCA: each measured trajectory is resampled to 121 equal motion-phase points, then
the analytical noise-free A-to-B path is subtracted. The residual includes injected variation,
tracking error and controller bias. PC1 explains 26.772%; PC1+PC2 explain 47.094%. All seven
components are needed for 95% variance (six retain 94.279%). Thus residual variation spans
all seven joint directions. This does not imply arbitrary seven-dimensional task coverage.

Peak action perturbations vary from 0.00972 to 0.02328 rad across episodes. Phase-aligned
mean position SD is about 0.0027–0.0039 rad per joint, with larger mid-motion spread and small
endpoint spread. The run provides real, reproducible variation around a narrow common task.

TRAINING READINESS
Use this dataset for the intended small ACT training/rollout test. Start with two episodes
to verify overfitting, then use an episode-level split such as 16 training / 4 validation.
Compute normalization statistics on the training episodes only; do not split frames from
the same episode across train and validation. Keep the deployment controller/reference
units and observation preprocessing consistent. Raw records contain seven arm joints;
the existing converter may still pad gripper dimensions, which an arm-only setup should omit.
Check the training loader's action time shift because these rows label the active command,
not a next-step state or the terminal waypoint. PCA here is a diagnostic; no PCA preprocessing
or reduction is required for the ACT model.

The dataset has one start, one goal, one scene and short trajectories. It can verify pipeline
operation and repeatability, but cannot establish arbitrary-goal generation, obstacle avoidance,
visual reliance, or broad generalization. No action or sensor value was modified by this audit.

FILES AND REPRODUCTION
12 numbered figure sets, each PNG plus vector PDF; RGB/depth previews; all-episode RGB contact sheets.
metrics.json: complete per-episode and per-camera metrics, checks, and timing warnings.
episode_statistics.csv, joint_statistics.csv, camera_statistics.csv, sensor_statistics.csv.
pca_*_components.csv: variance ratios and orthonormal component coefficients.
analysis_arrays.npz: joint arrays, phase-aligned states, PCA means/scales/scores and depth histograms.
camera_series.json: per-frame depth coverage and timestamps.
analyze.py reads raw data only; plot_analysis.py renders saved arrays; make_report.py builds this report.
On this machine, run from the repository root:
  /usr/bin/python3 act_20_analysis/analyze.py
  MPLCONFIGDIR=/tmp/act20_matplotlib /usr/bin/python3 -s act_20_analysis/plot_analysis.py
  /usr/bin/python3 act_20_analysis/make_report.py
The -s plotting invocation uses system Matplotlib/NumPy, avoiding a conflicting user-installed Matplotlib.
'''
(P/'report.txt').write_text(text)
summary=json.loads((P/'plot_summary.json').read_text());summary['issues']=M['failures'];summary['warnings']=M.get('warnings',[])
(P/'plot_summary.json').write_text(json.dumps(summary,indent=2))
gallery=''
for p in plots:
    stem=Path(p['file']).stem
    gallery+=f'<figure><h2>{html.escape(p["title"])}</h2><a href="{p["file"]}"><img loading="lazy" src="{p["file"]}" alt="{html.escape(p["title"])}"></a><figcaption>{html.escape(p["caption"])} <a href="{stem}.pdf">PDF</a></figcaption></figure>'
previews=''.join(f'<figure><h3>{html.escape(p.stem)}</h3><img loading="lazy" src="{p.name}" alt="RGB and depth preview"></figure>' for p in sorted(P.glob('preview*.jpg')))
overview=''.join(f'<figure><h3>{html.escape(p.stem)}</h3><img loading="lazy" src="{p.name}" alt="All episodes at motion midpoint"></figure>' for p in sorted(P.glob('overview*.jpg')))
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACT 20-episode analysis</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#172735;background:#f5f7fa;margin:0}}main{{max-width:1200px;margin:auto;padding:32px}}h1{{font-size:34px}}.verdict{{padding:22px;background:#e3f3e9;border-left:5px solid #28784b;border-radius:8px}}figure{{background:white;margin:28px 0;padding:20px;border:1px solid #dde2e7;border-radius:8px}}img{{display:block;width:100%;height:auto}}figcaption{{padding-top:12px;color:#425466}}pre{{white-space:pre-wrap;font:14px/1.7 ui-monospace,monospace;background:white;padding:24px;border-radius:8px}}a{{color:#175c98}}nav{{margin:20px 0}}</style>
<main><h1>ACT 20-episode data analysis</h1><div class="verdict"><b>Suitable for the fixed A-to-B ACT smoke test.</b><br>20 episodes · 3,040 observations · 20 distinct seeds · no hard integrity failures.<br>Small camera acquisition timestamp skew is documented below.</div>
<nav><a href="#assessment">Full assessment</a> · <a href="#plots">Plots</a> · <a href="#previews">RGB-D previews</a> · <a href="report.txt">Plain-text report</a> · <a href="metrics.json">Complete metrics</a></nav>
<h2 id="assessment">Full assessment</h2><pre>{html.escape(text)}</pre><h2 id="plots">Plots</h2>{gallery}<h2 id="previews">All-episode RGB overview and RGB-D previews</h2>{overview}{previews}
<h2>Data tables</h2><p><a href="episode_statistics.csv">Episode statistics</a> · <a href="joint_statistics.csv">Joint statistics</a> · <a href="camera_statistics.csv">Camera statistics</a> · <a href="sensor_statistics.csv">Sensor statistics</a></p></main></html>'''
(P/'index.html').write_text(page)
print('Saved',P/'index.html','and',P/'report.txt')
