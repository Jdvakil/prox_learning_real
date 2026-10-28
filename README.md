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
python record_dataset.py                   # -> recordings/<utc stamp>.h5
```

The arm moves through `POSITIONS` (top of `record_dataset.py`) while the robot state and the
newest packet of all 41 sensors are logged on a fixed 15 Hz clock. Recording waits until every
sensor has reported (`--allow-missing` to skip that) and prints tick timing, per-sensor packet
rate and packet age when it ends. Flags: `--pause` (hold at each waypoint, still logging),
`--tail`, `--dynamics`, `--no-gripper`, `--output`. Same h5 layout, without cameras.

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
