# prox_learning_real
Proximity learning implementation on the real robot: Franka FR3 + Franka Hand with 41 8x8
proximity sensors and two RealSense D435s (wrist, exocentric).

**Standalone lab deployment:** collection, the ACT model architecture, the trained model,
and evaluation are contained in this repository. No parent `prox_learning` installation
or sibling ACT checkout is needed. See [Lab PC setup](#lab-pc-setup) for dependency setup,
transfer, and hardware-free checks. `python utils.py bundle` creates a transferable
`dist/prox_learning_real_lab.tar.gz` with code, model weights and the local Franky source.

## Commands

Three scripts, run from this directory. `python <script>` lists its commands, and
`python <script> <command> -h` lists a command's flags.

```
python data.py cameras      list RealSense cameras, save snapshots, --live view / --record
python data.py record       move through POSITIONS, log robot + 40 sensors + cameras
python data.py pick         scripted pick of a block at a fixed spot (--set-pose first)
python data.py live         live browser view of sensors, cameras (--robot joints), no recording
python data.py teach        kinesthetic teaching (arm floats, move it by hand)
python data.py replay       replay taught demos on the stiff tracker, log rows
python data.py dashboard    offline HTML dashboard of recorded rows -> viz/

python model.py train-act-style  standalone ACT / PC-ACT trainer on converted pick data
python model.py eval pick        run a pick policy: --check-model, --offline, --sim or the robot
python model.py export pick      trained run (parent repo) -> models/

python utils.py check       Python packages + deployed model files, no hardware
python utils.py bundle      dist/prox_learning_real_lab.tar.gz for the lab PC
python utils.py robot-test  move the end effector 20 cm forward at 5% speed
python utils.py sim         rehearse teach / replay on the MuJoCo bench
```

The code behind them is in `prox_real/` (one module per command, e.g. `prox_real/record_dataset.py`,
`prox_real/eval_pick.py`; `prox_real/sim/` for the MuJoCo twin). `act_model/` is the
bundled ACT architecture, `models/` the deployed checkpoints, `tests/` the unit tests
(`python -m pytest tests`).

```bash
git submodule update --init --recursive    # franky 2.0 (build it per submodules/franky)
pip install pyrealsense2 "imageio[ffmpeg]" h5py numpy
```

## Lab PC: block pick evaluation (models/pick_act, models/pick_pcact)

Use `model.py eval pick`. The old A -> B evaluators (`eval act`, `eval pc-act`, models
`act_test_20` / `pc_act_test_20`) are retired: their data is deleted and on 2026-11-12 `eval act`
stopped at its first-target check on the current bench.

```bash
git pull && git lfs pull
python utils.py check
python model.py eval pick --model models/pick_act   --check-model --device cpu
python model.py eval pick --model models/pick_pcact --check-model --device cpu
# block on its spot (pick_pose.json), FCI on, workspace clear, hand on the e-stop
python model.py eval pick --model models/pick_act   --trials 1 --device cpu --wrist-serial 836612071883 --exo-serial 943222071556
python model.py eval pick --model models/pick_pcact --trials 1 --device cpu --wrist-serial 836612071883 --exo-serial 943222071556
python model.py eval pick --home     # after any abort
```

Details, safety stops and results: [Evaluate the pick policies on the arm](#evaluate-the-pick-policies-on-the-arm-gripper-included).

## Demonstrations: kinesthetic teach, then replay

```bash
python data.py cameras                                    # serials + one colour/depth snapshot each -> camera_check/
python data.py cameras --reset                            # same, USB power-cycle first (if "no frames" / stalled)
python data.py cameras --live                             # live RGB + aligned-depth dashboard for all connected cameras
python data.py cameras --live --reset --depth-max 2        # reset first; depth display scale 0–2 m
python data.py teach --home-gripper           # -> demos/<utc stamp>.h5
python data.py replay demos/*.h5 --wrist-serial <serial> --exo-serial <serial>
                                                     # -> dataset/rows/<utc stamp>_<demo>_r0/
```

The live camera dashboard opens every RealSense connected at startup concurrently. Each
tile shows RGB, aligned depth, serial/product name, measured frame rate, frame age and
valid-depth percentage. Invalid depth is black; near returns are red and far returns are
purple. The dashboard opens full screen; press `f` to toggle a resizable window. Quit with
`q`, Escape, the window close button, or Ctrl-C, including while cameras start or reset.
Camera startup runs in the background and times out after 90 seconds. Use
`--width`, `--height` and `--fps` to lower USB bandwidth when needed. Live mode
requires a desktop session and OpenCV GUI support (`opencv-python`); it does not write
snapshot files. Running without `--live` retains the existing per-camera snapshot check.

**teach**: the arm runs franky's joint impedance tracker with zero stiffness and zero damping, so
it floats under the robot's gravity compensation and you move it by hand. Keys in the terminal:
`space` start/stop a recording, `g` grasp/open, `h` home pose, `d` drop the recording, `q` quit.
Each recording saves q, dq, O_T_EE, gripper state and width at 100 Hz plus the gripper events.
Flags: `--damping` (Nms/rad, if the arm drifts: the skin adds mass the Franka model does not
know), `--friction-comp`, `--no-home`, `--grasp-force` (20 N), `--gripper-speed` (0.05 m/s),
`--torque-threshold` / `--force-threshold` (reflex while you push, 50 Nm / 80 N).

**replay**: per demo, the arm moves to its first pose and opens the gripper, you reset the scene
and step out of the camera views, then the recorded path is streamed to a stiff joint impedance
tracker (libfranka example gains) at 100 Hz. At each gripper event the path holds until the arm
has settled and the gripper finished. A 15 Hz loop logs the 41 sensors, robot state, commands
(joint reference, gripper 0 open / 255 grasp, commanded torque) and RGB + depth of both cameras.
You answer y (success) / n (failure) / d (discard). Flags: `--speed`, `--repeat`, `--max-error`
(stop if a joint lags the path by more, 0.15 rad), `--no-depth`, `--rgb-h5`, `--no-cameras`,
`--allow-missing`.

A row is the layout prox_learning's `scripts/convert_pact_place_to_act.py` converts for ACT /
PC-ACT: `trajectory.h5` (sim datagen layout `traj_0/obs/...`, `actions/joint_pos`, raw extras
under `traj_0/real/` including per-camera `depth_mm`, the replayed demo under `traj_0/teach/`),
`episode_00000000_wrist_camera.mp4`, `episode_00000000_exo_camera_1.mp4`, `result.json`.
See the docstring of `prox_real/kinesthetic.py`. About 8 MB per second of replay with depth.

## Rehearse in simulation

`python utils.py sim` (`prox_real/sim/twin.py`) runs teach / replay unchanged against a MuJoCo copy of the bench
(`prox_real/sim/lab_scene.py`: table, rails, two-drawer cabinet, block, both D435s; skin model
`sim_assets/franka_skin/model_hybrid_franka_hand.xml`, copied from prox_learning): a physics-backed fake
franky, the 41 sensors ray-cast and sent on the real UDP ports, and rendered D435 streams.

```bash
python utils.py sim rehearse                          # scripted teach, then replay -> sim_out/
python utils.py sim teach --viewer                    # your keys; drag the floating arm in the viewer
python utils.py sim replay sim_out/demos/<stamp>.h5 -- --speed 0.5
```

Edit `LAB` in `prox_real/sim/lab_scene.py` with the measured bench (cabinet, block, tripod, wrist mount).

## Pick up the block (scripted)

No model and no vision: the arm goes to one fixed grasp pose that you show it once.

```bash
# 1. Block at its spot. In Desk guiding mode, move the arm so the open fingers are around the
#    block; then FCI on, arm still:
python data.py pick --set-pose                        # -> pick_pose.json, no motion

# 2. Check the trajectory without hardware (durations, peak speeds, end points):
python data.py pick --plan-only
#    and run it on a simulated arm + 5 cm block (needs mujoco):
python data.py pick --sim --episodes 3               # -> recordings/pick_sim_<utc stamp>/

# 3. Collect (the arm moves; clear the workspace):
python data.py pick --episodes 10 --wrist-serial 836612071883 --exo-serial 943222071556
                                                      # -> recordings/pick_<utc stamp>/episode_0000/ ...
```

Each episode is one smooth joint trajectory streamed at 100 Hz to the joint impedance tracker. Recorded: start at HOME (`[0, 0, 0, -2.2, 0, 2.2, 0.7]`)
with the gripper open, a minimum-jerk joint move to 10 cm above the grasp pose, a straight
vertical descent, close the gripper (20 N, arm held still meanwhile), a straight 10 cm lift, a
1 s hold. Not recorded: lower the block back onto its spot (released `--place-gap`, 5 mm, above
the grasp height so it is never pressed into the table), open, rise, return HOME, so the next
episode finds the block in the same place. The straight lines use FR3 inverse kinematics seeded
from the joint angles saved by `--set-pose` (the elbow stays as you posed it), with minimum-jerk
timing, so every segment starts and ends at rest. All episodes are planned and checked (joint
limits, joint speed) before the arm moves. With the defaults: 2.9 s approach, 3.8 s descent,
3.8 s lift, peak 0.4 rad/s joint speed and 5 cm/s hand speed.

2026-11-10: the first version used franky `CartesianMotion` and every run aborted at the end of
the approach or descent with `cartesian_motion_generator_joint_velocity_discontinuity` (the
elbow was free and drifted, joint 6 ended at 3.30 instead of the posed 2.84). Hence the joint
trajectory.

2026-11-11: batches of 10 failed at the start of episode 2 (`tracking error 0.204 rad`, arm
sinking). Cause:
- franky ends a tracker with a damping-only `TorqueStopMotion`: zero stiffness for up to 2 s.
- The robot's gravity model lacks about 5 Nm on joint 2: mass at the hand (skin, sensors,
  camera). It shows as a 0.0087 rad lag while holding, at stiffness 600.
- So during each stop the arm sinks at about 0.1 rad/s, and episode 2 started 0.2 rad below
  HOME.

Now:
- The tracker stops within 0.3 s.
- Every episode starts with franky's position-controlled `JointMotion` to its start (HOME plus
  `--start-noise`, not recorded), and the batch ends at HOME.
- The tracker is not engaged unless the arm is within 0.03 rad of the episode start.

Setting the end-effector load (mass and centre of mass of hand + skin + camera) in Desk would
remove the lag and the sinking at the source. `--sim` models this load (1.2 kg uncompensated at
the hand) and franky's stop, and reproduced the crash (0.227 rad) before the fix.

Same row layout as the other recorders. Actions are the joint reference sent to the tracker at
each 15 Hz tick plus the gripper command (0 open, 1 close); `obs/extra/policy_phase` gives the step
(`start, approach, descend, grasp, lift, hold`). If the gripper is not holding the block after the
lift (width below `--min-width`, 3 mm), the row is saved with `clean_success: false`, the arm opens
at the lifted pose, rises and goes home without going down to the block again, and the batch
stops. libfranka's grasp reports success when the fingers close on nothing, so the width after
the grasp decides `gripper_closed_on_block`. Put the block back, or re-run `--set-pose` if the spot moved.
Flags: `--approach`, `--lift`, `--hold` (m, m, s), `--speed` (peak hand speed of the lines, m/s),
`--max-velocity` (peak joint speed of the joint moves, rad/s), `--max-error` (abort if a joint lags
its reference by more, 0.15 rad), `--jitter` (random x/y offset of the approach point, m),
`--grasp-force`, `--gripper-speed`, `--allow-missing`, plus the camera flags of `data.py record`.
The block must fit between the fingers (Franka Hand opens to 8 cm).

Collection noise (on by default) spreads the observations so the policy does not see one path
only. It changes the motion, not the recorded data: actions are still the reference actually
sent, observations are still raw. Image, depth and proximity noise belong in training augmentation.
- `--start-noise` (0.02 rad): Gaussian std per joint of each episode's start around HOME. The
  unrecorded `JointMotion` goes there, and the recording starts there.
- `--approach-noise` (0.005 m): Gaussian x/y/z offset of the point above the block.
- `--noise-std` (0.02 rad): bends the approach move with three smooth sine modes per joint.
  Position, velocity and acceleration of the bend are zero at both ends. The
  move is slowed if needed so the joint speed stays under `--max-velocity`.

Draws are clipped at 3 std and reproducible from `--seed`. Too large a std is refused:
3 std must stay under 0.15 rad and 0.02 m. The descent, grasp and lift are never perturbed;
the grasp pose stays exact. Each row stores `start_q`, `approach_offset_m`,
`approach_noise_modes_rad` and the three std values as `traj_0` attributes. Pass
`--start-noise 0 --approach-noise 0 --noise-std 0` for the old fixed path. In sim, 4 of 4
noisy episodes lifted the block 98-99 mm (max tracking error 0.013 rad, same as without noise).

`--sim` runs the same script against `prox_real/sim_franky.py`: a fake `franky` backed by MuJoCo
physics in real time (the parent repo's FR3 + hand model, copied into `sim_assets/franka_skin/`
so it also runs on the lab PC; a table and a cube centred at the
saved grasp pose, joint impedance with gravity compensation, force-limited fingers). No sensors
or cameras; it prints how far the block rose and where it was put back. `--sim-block` sets the
cube size (0 = no block, to exercise the failure path), `--sim-view` opens the MuJoCo viewer.
2026-11-10 with the lab pick_pose.json: 5 of 5 episodes lifted the block 99 mm, max tracking
error 0.009 rad. Without the place gap the block crept 2 mm per episode (it slipped in the
fingers, hit the table early and the hand, tilted 6.6 deg, shoved it); with it about 0.5 mm per
episode. The real drift is unknown: look at the block every ~10 episodes.

## Live dashboard

`data.py pick` and `data.py record` serve a live browser page while they
run and print its address (`http://localhost:8050`; from another machine use the robot PC's
hostname). It shows:
- **Joints:** each joint within its limits, the measured position next to the commanded target,
  a 20 s trace, the tracking error and the speed.
- **Status:** phase, episode clock and gripper width.
- **Cameras:** every open camera as a ~10 fps MJPEG stream.
- **Proximity sensors:** all 40 sensors as 8x8 heat maps grouped by link (block 1 or block 2;
  near = bright). A sensor with no packets is greyed out and a stale one (>0.5 s) is outlined;
  the header counts live sensors.

Missing pieces show as missing. Between episodes the joints are read straight from the arm.
The page opens in the desktop's browser by itself when the collector starts (before the arm
moves); `--no-browser` only serves it, `--no-dashboard` turns it off, `--dashboard-port` moves it.
Over ssh without a display nothing opens: use the printed address. It is `prox_real/live_dashboard.py`
plus `prox_real/dashboard_assets/live.html` (Python standard library HTTP server, no extra
packages, nothing runs until a browser connects). The collector's `base.sample()` hands it each
logged tick.

Check sensors and cameras before recording, without moving anything:

```bash
python data.py live --wrist-serial 836612071883 --exo-serial 943222071556
python data.py live --robot --gripper              # also joints and gripper, read-only
```

2026-11-11: `recordings/pick_20261111T220018Z` is the first clean real pick.
- **Pick:** gripper closed on the block at 24 mm and held it through the lift.
- **Tracking:** max error 0.010 rad, the same as the sim.
- **Cameras:** both at 30 fps with no dropped frames.
- **Proximity sensors:** none. All 40 were silent for the whole row (`--allow-missing`), so its
  proximity channels are empty. Check the dashboard's sensor count before collecting the batch.

2026-11-12: five noisy batches (`recordings/pick_20261112T*`, start σ 0.03 rad, approach σ
0.006 m, bend σ 0.03 rad):
- **Rows:** 48 recorded, 46 clean. Each lifted 95 mm with a 23.6–24.2 mm grip and a max tracking
  error of 0.0091–0.0098 rad. No dropped camera frames.
- **Failures:** `011644Z/0008`, where the wrist camera stopped (USB) and nothing was recorded; and
  `014342Z/0014`, where the gripper closed to 0 mm. The hand reached the grasp pose, and the wrist
  camera shows the block had drifted off its spot after 14 put-backs.
- **Repeated noise:** every batch used `--seed 0`, so the 46 clean rows hold only 20 distinct
  noise draws. Give each batch its own `--seed`.
- **Grasp offset:** at the grasp the hand is about −2 mm x, −1.3 mm y and −3 mm z off the saved
  pose in every episode. This is the joint 2 sag from the hand load missing in Desk.
- **Proximity:** 36 of 40 sensors return; `link5_front_sensor_0..3` are silent.
- **Bad config:** `011644Z/config.json` is not valid JSON (stray text at line 36). The h5
  attributes still hold its noise draws.

2026-11-12 work log, from the recordings to models ready for the arm:
1. **Reviewed the recordings:** results above.
2. **Converted to ACT format:** with the parent converter (`--with_proximity --prox_pool min`,
   320x240) into `act_style_data/real_pick_20261112/act`. This is the format the sim experiments
   train on.
3. **First training attempt, discarded:** `model.py train-act-style` (40 full-pass epochs, skin
   encoder from random weights) was not the sim recipe. It was stopped and is not used for results.
4. **Retrained with the sim recipe:**
   - Task `real_pick_20261112` added to `submodules/act/constants.py`.
   - `scripts/exp/train_real_pick.sh` is a copy of `train_pc_shelf_blind.sh`.
   - `EPOCHS=2000 EXP=ACT` and `EXP=PACT_READOUT`.
   - Launch arguments and skin config are identical to the `pc_shelf_blind` runs except for names.
   - Actions are 7 joint targets plus the gripper command; the state is 7 joints plus 2 fingers.
5. **Built the pick evaluator** (`model.py export pick`, `model.py eval pick`, section "Evaluate
   the pick policies on the arm"): HOME start, gripper, safety stops and the scripted put-back.
6. **Checked the evaluator against training:**
   - Inputs rebuilt from the raw rows equal the converted training files on all 46 rows.
   - The bundled model matches the parent `ACTPolicy` to within 4e-4 rad.
   - Final models: open-loop gripper agreement is 99.6-99.7% on 46 rows, and sim closed loop gave
     ACT 2/2 and PC-ACT 2/2.
7. **Tests:** `tests/test_eval_pick.py` adds 14 tests; 94 pass in total.
8. **Dashboard** (a claude.ai artifact): it shows the recordings, a video player, cumulative growth,
   a joint overlay and live training curves. A background agent re-uploads the curves every 100
   epochs, because 5 epochs take 2-3 s with this recipe.

## Waypoint recorder

```bash
python data.py record --wrist-serial <serial> --exo-serial <serial>   # -> recordings/<utc stamp>/

# Any number of named cameras (no wrist camera required):
python data.py record --camera wrist_camera <serial1> --camera exo_camera_1 <serial2> --camera exo_camera_2 <serial3>
```

Repeat `--camera NAME SERIAL` for each RealSense. Names and serials must be unique; names
use letters, digits, underscores and hyphens, starting with a letter or digit. The existing
`--wrist-serial` and `--exo-serial` flags still work and may be combined with `--camera`.
Use `python data.py cameras` to find the connected serials. Use `wrist_camera` and `exo_camera_1`
for the corresponding ACT converter views.

The arm moves through `POSITIONS` (top of `prox_real/record_dataset.py`) while the robot state and the
newest packet of all 41 sensors are logged on a fixed 15 Hz clock. All configured cameras are recorded at
their full rate (30 fps by default): every frame's depth goes into `trajectory.h5` (`traj_0/real/cameras/<name>/`,
with per-frame `t`, `frame_number`, and `tick_frame` = the frame used at each 15 Hz tick) and its
RGB into `<name>_<fps>fps.mp4`; the per-tick frame also goes into `episode_00000000_<name>.mp4` so
the row converts like a replay row. Cameras stream 3 s before the arm moves so auto white balance
settles (replay too). While recording, one physical core is reserved for the robot control thread
(`--control-cpu`, default last CPU, -1 = none) and video is encoded with x264 `ultrafast`: full load on
the control loop aborted motions with a discontinuity reflex with two cameras. The arm starts `--lead` s (1) after
logging starts and only once the encoders run, so rows begin with the arm still. If a camera gives no frames for 1 s or the motion fails, the arm is stopped
and the partial row is saved with `result.json` `clean_success: false` (the converter skips it,
the error is also in the `error` attribute of `traj_0`). Recording waits until every sensor has reported
(`--allow-missing` to skip that) and prints tick timing, per-sensor packet rate, camera fps,
dropped frames and writer backlog when it ends. Flags: `--pause` (hold at each waypoint, still
logging), `--tail`, `--dynamics`, `--no-gripper`, `--output` (row folder), `--no-cameras`,
`--cam-fps` (30), `--cam-width` / `--cam-height`, `--no-depth`, `--rgb-h5` (raw RGB of every
frame in the h5, large). Gripper only read (width), not commanded.

## ACT and PC-ACT on converted pick data

**Use the sim recipe for results.** The parent repo's `scripts/exp/train_real_pick.sh` trains the
converted data with `submodules/act/imitate_episodes.py`, exactly as the sim hidden-hazard
experiments do (task `real_pick_20261112` in `submodules/act/constants.py`):

```bash
cd /home/jaydv/code/prox_learning          # mlspaces env
EPOCHS=2000 EXP=ACT          ./scripts/exp/train_real_pick.sh   # -> submodules/act/ckpts/real_pick_20261112/<run>/
EPOCHS=2000 EXP=PACT_READOUT ./scripts/exp/train_real_pick.sh
```

That recipe uses batch 8, chunk 50, lr 1e-5 and KL weight 10. An epoch samples one random frame
per training episode, and the split is a random 80/20 by episode. PC-ACT fine-tunes the pretrained
corridor_v5 surface encoder. Each run takes about 1 h for 2000 epochs when both share the GPU.
`model.py train-act-style` (below) is a separate standalone trainer: it trains the encoder from
random weights and uses full-pass epochs, so its numbers do not compare to the sim runs.

`python model.py train-act-style` trains either arm on the ACT-style HDF5s that the parent repo's
converter writes. Both arms run the same code, split, normalization and schedule, so the only
difference is the proximity input. `--arm act` uses cameras and joints. `--arm pc-act` adds the
40-sensor skin through the surface encoder: a conv stem per 8x8 frame, then a temporal
transformer over eight causal 15 Hz snapshots, giving 40 x 128 readout tokens. The encoder is
trained end to end from random weights.

```bash
# Parent repo root: stage the batch rows flat, then convert (cameras 240x320, proximity min-pooled)
mkdir -p act_style_data/real_pick_20261112/rows
for ep in submodules/prox_learning_real/recordings/pick_*/episode_*; do
  b=$(basename $(dirname $ep)); ln -sfn $(realpath $ep) act_style_data/real_pick_20261112/rows/${b#pick_}_$(basename $ep)
done
python -m scripts.convert_pact_place_to_act --src act_style_data/real_pick_20261112/rows \
  --dst act_style_data/real_pick_20261112/act --with_proximity --prox_pool min

# This repo: train both arms (they fit on the 24 GB GPU together, about 2 min per epoch each)
P=../../act_style_data/real_pick_20261112
python model.py train-act-style --arm act    --data $P/act --rows $P/rows --run-dir training_runs/pick_act_20261112 --epochs 40
python model.py train-act-style --arm pc-act --data $P/act --rows $P/rows --run-dir training_runs/pick_pcact_20261112 --epochs 40
```

An epoch is one pass over every training frame, in batches of 16 (562 steps for the 2026-11-12
data). The 8 validation episodes come from 3 held-out noise draws: every episode that shares a
start pose with one of them is also held out, so validation start poses never appear in training.
Normalization statistics come from the training episodes only. Actions are the 7 joint
references plus the gripper command; the state is the 7 joints plus the two finger positions.
Each run writes `metrics.jsonl` (one line per epoch: train loss/L1/KL, validation L1,
validation arm error in rad, validation gripper error), `progress.json`, `dataset_stats.npz`
and `policy_best.ckpt` (PC-ACT also writes `prox_encoder_best.pt`, from the same epoch). Low
validation L1 is offline imitation error, not task success on the robot.

## Evaluate the pick policies on the arm (gripper included)

`python model.py eval pick` runs an ACT or PC-ACT pick policy trained with the sim recipe above.
It also commands the gripper.

```bash
# Desktop, training env (the run's dataset_stats.pkl is a numpy 2 pickle): make portable bundles
C=../act/ckpts/real_pick_20261112; D=../../act_style_data/real_pick_20261112/act
python model.py export pick --source $C/real_pick_20261112_ACT_s0_bs8_cs50_lr1e-5_e2000 \
  --destination models/pick_act --data-dir $D
python model.py export pick --source $C/real_pick_20261112_PACT_READOUT_s0_bs8_cs50_lr1e-5_e2000 \
  --destination models/pick_pcact --data-dir $D

# Lab PC, no hardware: load check, open-loop check on recorded rows, closed loop in MuJoCo
python model.py eval pick --model models/pick_act --check-model
python model.py eval pick --model models/pick_act --offline recordings/pick_20261112T022139Z
python model.py eval pick --model models/pick_act --sim --trials 2

# Robot: put the block on its spot (pick_pose.json), FCI on, clear workspace
python model.py eval pick --model models/pick_act   --trials 3 --wrist-serial 836612071883 --exo-serial 943222071556
python model.py eval pick --model models/pick_pcact --trials 3 --wrist-serial 836612071883 --exo-serial 943222071556
python model.py eval pick --home        # after an abort: position-controlled move HOME, gripper untouched
```

**No GPU needed.** With no CUDA the evaluator runs on CPU (or force it with `--device cpu`). One
query takes about 0.2 s on CPU (Threadripper 2970WX). Each chunk is timed from the moment its
observation was taken, so that delay only skips the chunk's first 3 steps. CPU-only sim runs passed
for both models (ACT: block rose 96 mm; PC-ACT: 95 mm). Don't use `--temporal-agg` on CPU: it
queries every 67 ms tick and can't keep up.

**Inputs** are built exactly as training saw them. On all 46 rows, inputs rebuilt from the raw
recordings match the converted training files exactly (zero difference in joints, skin, images
and actions):
- The 7 measured joints plus both fingers at half the gripper width.
- Both cameras resized to 320x240 with `INTER_AREA`.
- PC-ACT: the newest packet of every skin sensor. It uses block 1 in metres, with no return = 4 m,
  kept as the last 8 ticks at 15 Hz and encoded with `encode_pooled_history`, as in the training loader.

The bundled model matches the parent repo's training-time `ACTPolicy` to within 4e-4 rad on the
same inputs.

**Control** follows the sim protocol:
- The policy runs at 15 Hz and plays each 50-step chunk before it queries again
  (`--exec-horizon N` and `--temporal-agg` change this).
- Targets are interpolated to 100 Hz for the joint impedance tracker and rate limited to 0.5 rad/s.
- A gripper output above 0.5 closes the hand with the collection grasp (20 N, any width); below
  0.5 opens it.

**Every trial starts at HOME** with the gripper open. Any of these stops the arm with the short
damping stop and ends the batch:
- a target within 0.05 rad of a joint limit;
- the hand more than `--floor-margin` (15 mm) below the saved grasp height;
- the hand outside the HOME-to-grasp box plus `--box-margin` (10 cm);
- a policy target more than 0.15 rad from the measured arm;
- tracking error above `--max-error` (0.10 rad);
- no new chunk for 1 s.

After an abort nothing scripted moves the arm. Run `--home` once you have checked it.

**A trial ends early** once the block has been held 8 cm above the grasp height for 1 s;
otherwise it ends after `--duration` (20 s). Success = gripper holding (width above 3 mm) and
lifted. The script then puts the block back with the `data.py pick` segments (`--no-put-back`
skips this). Results go to `evaluations/pick_<model>_<utc>/trial_NNNN/`: `rollout.h5` holds
joints, references, policy actions, chunks and latency; `result.json` holds success, gripper
events, depth reached and tracking error. `--save-images` also stores the policy's RGB input.

Final models, 2026-11-12. The bundles are `models/pick_act` (ACT, best validation loss 0.0477 at
epoch 1984) and `models/pick_pcact` (PC-ACT, 0.0458 at epoch 1968):
- **Matches training:** the bundled models are within 1e-3 rad of the parent repo's
  training-time `ACTPolicy` on the same inputs.
- **Open loop on all 46 rows** (`--offline`): the gripper output agrees with the recorded command
  on 99.6% (ACT) and 99.7% (PC-ACT) of ticks. Every row closes, on average within a quarter of a
  tick of the recorded close. The mean arm error over a chunk is 0.016 rad (ACT) and 0.015 rad
  (PC-ACT).
- **Closed loop in sim** (`--sim`, 24 mm block, replayed camera video): ACT 2/2 and PC-ACT 2/2
  grasped, lifted 95-103 mm, held, and the block was put back within 7.5 mm.
  - The hand never went more than 2.6 mm below the saved grasp height (the floor stop is at 15 mm).
  - Max tracking error was 0.037 rad.
  - The epoch-500 checkpoints had reached 10 mm below grasp height; the final ones don't.
- The 80/20 split is random, so validation episodes can share their noise draw with training
  episodes (see the data notes), and the validation loss is somewhat optimistic.

**Does PC-ACT use the skin?** On the robot it gets all 40 sensors live (the last 8 ticks,
through the fine-tuned encoder). On 770 chunks from 16 recorded episodes:
- Clearing the skin to "no return" moves its arm prediction by 0.0032 rad on average (at most 0.016).
- Its error against the recording goes from 0.0087 to 0.0090 rad.
- Swapping in another episode's skin moves the prediction by 0.0014 rad.

So the skin is wired in and has a small effect, and the actions come almost entirely from cameras
and joints. That is expected with no obstacles: the skin sees nearly the same table and block in
every episode. The `--sim` checks run PC-ACT with no skin at all (the twin has no sensors).

`--sim-video` renders each sim trial to `trial_NNNN/sim.mp4` (side | top view, offscreen through
`MUJOCO_GL=egl`). `sim_state.npz` keeps the 10 Hz MuJoCo state, so a trial can be rendered again later.

The pick has no obstacles, so PC-ACT and ACT should behave about the same here. This is a test
of the pipeline and the policy, not of the skin.

Cameras: on the robot PC a D435 gives no frames on a fresh start until it is USB power-cycled
(2026-10-28: `cameras.py` and `record_dataset.py` both failed with "no frames" / "Frame didn't
arrive" until `cameras.py --reset`). `data.py record`, `data.py pick` and `data.py replay` therefore
reset the cameras they use before starting them (about 5 s), and if one still gives no frames
they reset and retry once. `--no-camera-reset` skips the first reset.

## Lab PC setup

Only this repository is needed on the lab PC. Transfer it one of two ways:

```bash
python utils.py bundle                    # -> dist/prox_learning_real_lab.tar.gz (code, models, Franky source)
tar -xzf prox_learning_real_lab.tar.gz && cd prox_learning_real          # on the lab PC
```

The archive excludes recordings, analysis output, Git metadata, credentials, installed
binaries and caches, and contains the real checkpoints. Or clone with Git; the checkpoints
use Git LFS:

```bash
git lfs install
git submodule update --init --recursive
git lfs pull
```

Python 3.10 or 3.11, in the environment with the working Franky build:

```bash
python -m pip install -r requirements-collection.txt   # collection only
python -m pip install -r requirements-eval.txt         # + pick evaluation (Torch 2.7.1 / torchvision 0.22.1, mujoco for --sim)
python -m pip install -r requirements-training.txt     # + training and W&B
```

Keep a working Franky installation if it exposes `JointImpedanceTracker`. Otherwise build
the included source after setting up the robot-compatible libfranka, Eigen3 (>= 3.4),
ruckig (0.17.3), CMake / compiler and pybind11:

```bash
python -m pip install --no-build-isolation ./submodules/franky
```

Hardware-free checks: `python utils.py check` (`--offline-only` before the camera and robot
drivers are installed) and `python model.py eval pick --model models/pick_act --check-model
--device cpu` (and `models/pick_pcact`), which load the bundle and predict one (50, 8) chunk from
dummy input without moving the robot. New models come from `model.py export pick` (section
"Evaluate the pick policies on the arm"); copy the whole model folder and pass it as `--model`.

First motion after setup: `python utils.py robot-test` moves the end effector 20 cm along its
own x axis at 5% of the velocity / acceleration / jerk limits (`--distance`, `--host`). Clear
the space in front of the hand first.

## Sensors

`prox_real/sensor_layout.py` holds the ports, sensor counts and the mapping to the sim camera names.

```
link1        port 5004   7 sensors
link2        port 5005   7 sensors
link3        port 5006   5 sensors
link5_small  port 5007   4 sensors   (sim link5_front)
link5_large  port 5008   6 sensors   (sim link5_back)
link6        port 5009   6 sensors
link4        port 5010   5 sensors
end_effector port 5011   1 sensor    (sim link7_hand)
```

Packet: 1 byte sensor id + 128 uint16 mm = two row-major 8x8 blocks. Block 1 = confident
zones (4000 = no target), block 2 = block 1 plus low-confidence fills.

## View

In the prox_learning repo: `python scripts/real_skin_viz.py <recording or row>/trajectory.h5`
writes a 3D replay (skin model moving as recorded, zones projected from each sensor).

Dashboard: `python data.py dashboard <row>` (a folder holding `trajectory.h5`, e.g.
`recordings/pick_<utc stamp>/episode_0000`) writes `viz/<row name>_dashboard.html`, one
offline file: pass/warn tiles (blocked camera check included), synced camera and sensor replay, joint
position and velocity, nearest return per link, packet rates and stuck zones. Needs h5py, numpy, opencv.