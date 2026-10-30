# prox_learning_real
Proximity learning implementation on the real robot: Franka FR3 + Franka Hand with 41 8x8
proximity sensors and two RealSense D435s (wrist, exocentric).

```bash
git submodule update --init --recursive    # franky 2.0 (build it per submodules/franky)
pip install pyrealsense2 "imageio[ffmpeg]" h5py numpy
```

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
reference active at that observation, not the final waypoint. The gripper is opened once
and held open; `--no-gripper` skips opening/reading and requires it to be kept open manually.
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

Each batch contains `config.json` and `episode_0000/`, `episode_0001/`, etc. Each episode
has the usual `trajectory.h5`, camera videos and `result.json`, plus `reference_plan.npz`.
HDF5 extras include `real/nominal_q`, `real/noise_q`, `real/dq_ref`, `real/clock`, the noise
coefficients, and `real/control/{t,q_ref,dq_ref}` for every reference actually sent at 100 Hz.
`actions/joint_pos` and `real/target_q` contain the corresponding 15 Hz references. No action
time shift is applied here; account for any shift performed by the training loader.

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
