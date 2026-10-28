# prox_learning_real
Proximity learning implementation on the real robot: Franka FR3 + Franka Hand with 41 8x8
proximity sensors.

## Record

```bash
git submodule update --init --recursive    # franky (build it per submodules/franky)
python record_dataset.py                   # -> recordings/<utc stamp>.h5
```

The arm moves through `POSITIONS` (top of `record_dataset.py`) while the robot state and the
newest packet of all 41 sensors are logged on a fixed 15 Hz clock. Recording waits until every
sensor has reported (`--allow-missing` to skip that) and prints tick timing, per-sensor packet
rate and packet age when it ends. Flags: `--pause` (hold at each waypoint, still logging),
`--tail`, `--dynamics`, `--no-gripper`, `--output`.

The h5 uses the sim datagen layout (`traj_0/obs/agent/qpos`, `obs/extra/tcp_pose`,
`obs/proximity/<sim sensor>` as (T, 1, 8, 8) metres, `actions/joint_pos`, ...); raw packets,
packet age and timestamps are under `traj_0/real/`. See the docstring of `record_dataset.py`.

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

In the prox_learning repo: `python scripts/real_skin_viz.py <recording>.h5` writes a 3D replay
(skin model moving as recorded, zones projected from each sensor).
