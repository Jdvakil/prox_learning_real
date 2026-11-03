# prox_learning_real
Proximity learning implementation on the real robot: Franka FR3 + Franka Hand with 41 8x8
proximity sensors and two RealSense D435s (wrist, exocentric).

**Standalone lab deployment:** collection, the ACT model architecture, the trained model,
and evaluation are contained in this repository. No parent `prox_learning` installation
or sibling ACT checkout is needed. See [LAB_SETUP.txt](LAB_SETUP.txt) for dependency setup,
transfer, and hardware-free checks. `python make_lab_bundle.py` creates a transferable
`dist/prox_learning_real_lab.tar.gz` with code, model weights and the local Franky source.

```bash
git submodule update --init --recursive    # franky 2.0 (build it per submodules/franky)
pip install pyrealsense2 "imageio[ffmpeg]" h5py numpy
```

## Lab PC: first ACT rollout after pulling

Run these commands from the `prox_learning_real` directory in your working Franky
Python environment. No parent `prox_learning` checkout is required.

1. Fetch the model weights and install the evaluation dependencies:

   ```bash
   git lfs pull
   python -m pip install -r requirements-eval.txt
   ```

   Keep the working local Franky installation that exposes `JointImpedanceTracker`.
   If it is missing, follow the build instructions in [LAB_SETUP.txt](LAB_SETUP.txt).

2. Check dependencies and load the model without connecting to the robot or cameras:

   ```bash
   python check_lab_setup.py
   python eval_act_real.py --check-model --device cpu
   ```

   The setup check should pass. The model check should report `"finite": true`,
   `"action_shape": [50, 8]`, and `"hardware_connected": false`. If either check
   fails, resolve the reported error before starting a rollout. A Git LFS pointer
   error means the actual checkpoint still needs to be fetched with `git lfs pull`.

3. Enable FCI, clear the trained motion and reset path, and run one evaluation:

   ```bash
   python eval_act_real.py \
     --wrist-serial 836612071883 \
     --exo-serial 943222071556 \
     --output eval_rollouts/trial_01
   ```

   **This command moves the arm to training start A, then executes the policy.**
   The serials must correspond to the original wrist and exocentric views. The
   evaluator does not connect to or command a gripper. Use Ctrl-C to abort.
   Choose a new output directory for each trial; existing directories are rejected.

4. Inspect the result:

   ```bash
   cat eval_rollouts/trial_01/result.json
   ```

   Check `success`, `error`, `final_goal_error_rad`, `final_max_speed_rad_s`, and
   `query_latency_s`. Success requires goal error at most 0.02 rad and joint speed
   at most 0.01 rad/s throughout the final 0.5 seconds. `rollout.h5` contains the
   measured states, camera inputs, policy actions and actual controller references;
   `config.json` records the model and execution settings. Inspect a failed trial's
   error and logs before repeating it. The stopping motion is excluded from the
   recorded evaluation.

See [Evaluate the trained ACT policy on the real arm](#evaluate-the-trained-act-policy-on-the-real-arm)
below for checkpoint selection, control limits and inference options.

## Demonstrations: kinesthetic teach, then replay

```bash
python cameras.py                                    # serials + one colour/depth snapshot each -> camera_check/
python cameras.py --reset                            # same, USB power-cycle first (if "no frames" / stalled)
python kinesthetic.py teach --home-gripper           # -> demos/<utc stamp>.h5
python kinesthetic.py replay demos/*.h5 --wrist-serial <serial> --exo-serial <serial>
                                                     # -> dataset/rows/<utc stamp>_<demo>_r0/
```

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
See the docstring of `kinesthetic.py`. About 8 MB per second of replay with depth.

## Rehearse in simulation

`sim/twin.py` runs `kinesthetic.py` unchanged against a MuJoCo copy of the bench
(`sim/lab_scene.py`: table, rails, two-drawer cabinet, block, both D435s; skin model from
prox_learning `assets/robots/franka_skin/model_hybrid_franka_hand.xml`): a physics-backed fake
franky, the 41 sensors ray-cast and sent on the real UDP ports, and rendered D435 streams.

```bash
python sim/twin.py rehearse                          # scripted teach, then replay -> sim_out/
python sim/twin.py teach --viewer                    # your keys; drag the floating arm in the viewer
python sim/twin.py replay sim_out/demos/<stamp>.h5 -- --speed 0.5
```

Edit `LAB` in `sim/lab_scene.py` with the measured bench (cabinet, block, tripod, wrist mount).

## Waypoint recorder

```bash
python record_dataset.py --wrist-serial <serial> --exo-serial <serial>   # -> recordings/<utc stamp>/

# Any number of named cameras (no wrist camera required):
python record_dataset.py --camera wrist_camera <serial1> --camera exo_camera_1 <serial2> --camera exo_camera_2 <serial3>
```

Repeat `--camera NAME SERIAL` for each RealSense. Names and serials must be unique; names
use letters, digits, underscores and hyphens, starting with a letter or digit. The existing
`--wrist-serial` and `--exo-serial` flags still work and may be combined with `--camera`.
Use `python cameras.py` to find the connected serials. Use `wrist_camera` and `exo_camera_1`
for the corresponding ACT converter views.

The arm moves through `POSITIONS` (top of `record_dataset.py`) while the robot state and the
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

### Simple start-to-goal ACT smoke test

`record_act_waypoints.py` records one A -> B motion per row, with the reset to A outside
the recording. It defaults to two episodes with the same start and goal; each seed produces
a different smooth perturbation of the path. Review `START` / `GOAL` in the script for the
workspace, or supply seven joint angles in radians using `--start` / `--goal`.

```bash
# Validate and save the 20 reference plans without connecting to any hardware:
python record_act_waypoints.py --dry-run --episodes 20 --output /tmp/act_plans

# Collect two episodes first (default); change to --episodes 20 for the batch:
python record_act_waypoints.py --episodes 2 --seed 0 \
  --wrist-serial <serial1> --exo-serial <serial2> --output recordings/act_smoke
```

The default motion takes 8 seconds, with 1 second of lead-in and 1 second holding B.
A minimum-jerk reference runs through the joint impedance tracker at 100 Hz; observations
and ACT action labels are sampled at 15 Hz. Actions contain the actual noisy absolute joint
reference active at that observation, not the final waypoint. Only the seven arm joints
are used; there is no gripper connection, command, polling, or saved gripper state.
The existing ACT converter still pads its fixed observation/action dimensions with zeros
for missing gripper fields; an arm-only training setup should select the first seven entries.
Each row can be processed by the existing ACT converter using the usual camera names.
Repeat `--camera NAME SERIAL` for more cameras; the converter's supported camera names
still apply. RGB-D/proximity storage flags match `record_dataset.py`.

`--noise-std 0.01` is the Gaussian coefficient scale in radians for three low-frequency sine
modes per joint. A smooth envelope makes perturbation, velocity and acceleration zero at
the endpoints. This adds variation to executed commands and observations. Set it to 0 for
the deterministic baseline. Episode seeds are `--seed + episode index`; use a different
base seed for another batch. The noisy references are the training targets, so this smoke
test learns the perturbed demonstration distribution, not a clean expert correction policy.

Before touching hardware, every plan is checked against FR3 joint limits and configurable
reference speed, acceleration and perturbation bounds. These checks do not test collisions;
both the reference and the reset path need a clear workspace. Tracking errors, camera stalls
and recording failures abort the batch and mark the partial row `clean_success: false`.
Recording ends while the tracker still holds B; cameras are finalized before controller
shutdown, so the damping-only shutdown motion is excluded from training episodes.

Each batch contains `config.json` and `episode_0000/`, `episode_0001/`, etc. Each episode
has the usual `trajectory.h5`, camera videos and `result.json`, plus `reference_plan.npz`.
HDF5 extras include `real/nominal_q`, `real/noise_q`, `real/dq_ref`, `real/clock`, the noise
coefficients, and `real/control/{t,q_ref,dq_ref}` for every reference actually sent at 100 Hz.
`actions/joint_pos` and `real/target_q` contain the corresponding 15 Hz references. No action
time shift is applied here; account for any shift performed by the training loader.

### Evaluate the trained ACT policy on the real arm

`eval_act_real.py` loads `models/act_test_20/policy_best.ckpt`, portable `dataset_stats.npz`,
and saved training/image metadata. Its model architecture is bundled in `act_model/`.
There is no runtime dependency on the parent repository, an ACT installation, or the
converted training dataset. Legacy directories containing `dataset_stats.pkl` are also
supported, but new deployments should use the portable exported model folder.
It uses the training camera order (`exo_camera_1`, `wrist_camera`), 320x240 RGB input,
ImageNet image normalization, and the saved joint/action statistics. Nine-dimensional
state inputs receive two zero padding entries; only the first seven predicted arm positions
are sent to the robot. It never connects to a gripper. This is vanilla RGB + joint-state ACT;
depth and proximity are not model inputs, and PACT checkpoints are rejected.

Install `requirements-eval.txt` in an environment with the local Franky build exposing
`JointImpedanceTracker`. Use `python check_lab_setup.py` to check dependencies, then
`python eval_act_real.py --check-model --device cpu` to load the model and run a dummy
prediction without connecting to hardware. See `LAB_SETUP.txt` for the standalone setup.

```bash
# No hardware connection: checkpoint and image-pipeline check on a recorded episode.
python eval_act_real.py --offline-row recordings/act_test_20/episode_0000 \
  --device cpu --output eval_rollouts/offline_ep0

# One real rollout: resets to training start A, runs the policy, and evaluates arrival at B.
python eval_act_real.py \
  --wrist-serial 836612071883 --exo-serial 943222071556 \
  --output eval_rollouts/real_trial_01
```

`--ckpt-dir` / `--ckpt-name` select another checkpoint. Camera serials must match the physical
training views; `--camera NAME SERIAL` is also supported. `--no-reset` requires the arm
already to be within the start tolerance. Default A/B match the collector; `--start` / `--goal`
take seven joint angles. `--duration` defaults to 10.1 seconds, including the learned lead-in
and goal hold. The return to A is a Franky `JointMotion` at `--dynamics 0.05`; it is outside
the rollout. Check that this reset path and the trained motion have clear space.

The main loop reads observations and selects absolute joint targets at 15 Hz. Inference
runs in one background worker. The first chunk is prepared at A before torque control
starts; subsequent queries begin up to `--prefetch-steps 8` ticks before the old chunk
expires. A delayed prediction keeps its original observation/tick origin: already elapsed
elements are skipped, so an old chunk is never restarted from its first action. A 50-step
checkpoint therefore normally starts a new query 42 ticks after the previous query.
`--query-every N` requests more frequent replanning; `--temporal-agg` queries whenever the
worker is available and combines overlapping predictions, favoring recent ones with
`--aggregation-decay`. Actual query ticks are logged; compute throughput can limit them.
The simpler offline mode executes fixed chunks without real-time prefetch scheduling;
its teacher-forced prediction error is not a real-robot success rate.

A separate 100 Hz Franky joint impedance tracker follows a critically damped reference
with bounded velocity (0.15 rad/s) and acceleration (0.5 rad/s²). This smooths the policy's
15 Hz position commands. The dynamics factor applies to the reset, while these explicit
limits apply to streamed references. Commands outside the FR3/task joint envelope or more
than 0.12 rad from measured joints are rejected. The default task envelope is the A/B joint
range plus 0.08 rad, intersected with joint limits. Camera age over 0.25 seconds, missing
policy updates, inference delays over 0.5 seconds, excessive tracking error (0.15 rad),
invalid outputs, or exhausted chunks abort and save a failed rollout. The controller stops
on exceptions or Ctrl-C. These checks do not provide collision avoidance.

Success requires measured goal error at most 0.02 rad and maximum measured joint velocity
at most 0.01 rad/s throughout the last 0.5 seconds. This is measured while the tracker still
holds position; shutdown motion is excluded. Every run creates a new output directory:

- `config.json`: checkpoint, training configuration and resolved image/camera settings.
- `rollout.h5`: observation times, measured joints/velocities, actual smoothed references,
  selected policy actions, RGB inputs, camera frame numbers/ages, query ticks and complete
  predicted chunks, and the actual 100 Hz reference trace.
- `result.json`: success/error, goal error, query latency and execution settings. In offline
  mode it reports action prediction errors and explicitly leaves rollout success unset.

The original ACT loader shifts real-robot labels by one observation. Offline output reports
errors against both the active recorded reference and this previous-action target. Live
execution sends absolute predicted positions directly; it does not apply an extra shift.
Legacy normalization statistics pickled with NumPy 2 require a compatible NumPy runtime.
The bundled NPZ statistics avoid that restriction. `export_act_model.py` packages updated
checkpoints, statistics and metadata into another self-contained model folder.

Cameras: on the robot PC a D435 gives no frames on a fresh start until it is USB power-cycled
(2026-10-28: `cameras.py` and `record_dataset.py` both failed with "no frames" / "Frame didn't
arrive" until `cameras.py --reset`). `record_dataset.py` and `kinesthetic.py replay` therefore
reset the cameras they use before starting them (about 5 s), and if one still gives no frames
they reset and retry once. `--no-camera-reset` skips the first reset.

## Sensors

`sensor_layout.py` holds the ports, sensor counts and the mapping to the sim camera names.

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

Dashboard: `python dashboard_row.py recordings/<utc stamp>` writes `viz/<utc stamp>_dashboard.html`, one
offline file: pass/warn tiles (blocked camera check included), synced camera and sensor replay, joint
position and velocity, nearest return per link, packet rates and stuck zones. Needs h5py, numpy, opencv.
