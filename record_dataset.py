"""Move the Franka through joint waypoints and log the robot and all 41 proximity sensors at 15 Hz.

The arm runs in a background thread while the main loop samples on a fixed 15 Hz clock, so the
log covers the motion itself, not only the stops. A listener thread keeps the newest packet of
every sensor; each tick stores that packet and how old it is.

The HDF5 file uses the sim datagen layout (MolmoSpaces trajectories_*.h5) so the prox_learning
converters and viewers read it the same way:

  traj_0/obs/agent/qpos            (T, 2000) uint8  JSON {"arm": [7], "base": [], "gripper": [2]}
  traj_0/obs/agent/qvel            (T, 2000) uint8  same keys
  traj_0/obs/extra/tcp_pose        (T, 7)  [x, y, z, qw, qx, qy, qz] of O_T_EE
  traj_0/obs/extra/robot_base_pose (T, 7)  identity (base frame = world)
  traj_0/obs/extra/policy_phase    (T,)    index of the waypoint being executed
  traj_0/obs/proximity/<sim name>  (T, 1, 8, 8) float32 m, block 1 (confident zones),
                                   no target = 4.0 m
  traj_0/actions/joint_pos         (T, 2000) uint8  JSON {"arm": waypoint target, "gripper": [0.0]}
  traj_0/real/...                  raw extras: both 8x8 blocks in mm, packet age, fresh flag,
                                   timestamps, O_T_EE, elbow, gripper width

<sim name> is the camera in assets/robots/franka_skin/model_hybrid_franka_hand.xml
(sensor_layout.sim_name). Each packet is 1 id byte + 128 uint16 mm: two row-major 8x8 blocks,
block 1 confident zones only (4000 = no target), block 2 the same plus low-confidence fills.

Requires franky (built from submodules/franky) and h5py.
"""

import argparse
import json
import select
import socket
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from sensor_layout import LINKS, NO_TARGET_MM, SENSORS_PER_LINK, sim_name

GROUP = "239.0.0.1"
RATE_HZ = 15.0
ZONES = 8
JSON_ROW = 2000                  # bytes per JSON row, as in the sim files
NO_TARGET_M = NO_TARGET_MM / 1000.0
SENSORS = [(name, sensor_id) for name, count in SENSORS_PER_LINK.items() for sensor_id in range(count)]

# Joint targets [rad] from the franky waypoint example. Edit these for the workspace.
POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7],
        [0.2, -0.1, 0.3, -2.0, 0.1, 1.9, 0.6],
        [0.4, -0.3, 0.1, -2.3, 0.0, 2.2, 0.5],
        [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7],
    ],
    dtype=np.float64,
)


def open_sensor_sockets():
    membership = struct.pack("4sL", socket.inet_aton(GROUP), socket.INADDR_ANY)
    sockets = {}
    for name, port in LINKS.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sockets[sock] = name
    return sockets


class SensorListener(threading.Thread):
    """Keeps the newest packet of every (link, sensor id) with its arrival time and a counter."""

    def __init__(self, sockets):
        super().__init__(daemon=True)
        self.sockets = sockets
        self.lock = threading.Lock()
        self.latest = {}             # (link, id) -> (depth (128,) uint16, monotonic time, count)
        self.bad = 0
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            readable, _, _ = select.select(list(self.sockets), [], [], 0.1)
            for sock in readable:
                data, _ = sock.recvfrom(65535)
                now = time.monotonic()
                if len(data) != 1 + 2 * ZONES * ZONES * 2:
                    self.bad += 1
                    continue
                key = (self.sockets[sock], int(data[0]))
                depth = np.frombuffer(data[1:], dtype=np.uint16).copy()
                with self.lock:
                    count = self.latest[key][2] + 1 if key in self.latest else 1
                    self.latest[key] = (depth, now, count)

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def missing(self):
        with self.lock:
            return [f"{name} sensor {sensor_id}" for name, sensor_id in SENSORS if (name, sensor_id) not in self.latest]


class GripperReader(threading.Thread):
    """Polls the Franka Hand width; Gripper.width blocks on the gripper's own state stream."""

    def __init__(self, host):
        super().__init__(daemon=True)
        from franky import Gripper

        self.gripper = Gripper(host)
        self.width = float("nan")
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.width = float(self.gripper.width)
            except Exception:
                self.width = float("nan")
                time.sleep(0.05)


class MotionRunner(threading.Thread):
    """Moves through the waypoints one after another, pausing at each."""

    def __init__(self, robot, positions, pause):
        super().__init__(daemon=True)
        self.robot = robot
        self.positions = positions
        self.pause = pause
        self.phase = 0
        self.error = None
        self.done = threading.Event()

    def run(self):
        from franky import JointMotion

        try:
            for index, target in enumerate(self.positions):
                self.phase = index
                print(f"waypoint {index}: moving to {target.tolist()}")
                self.robot.move(JointMotion(target.tolist()))
                time.sleep(self.pause)
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()


def json_row(value):
    raw = json.dumps(value).encode("utf-8")
    if len(raw) >= JSON_ROW:
        raise ValueError(f"JSON row too long ({len(raw)} bytes)")
    row = np.zeros(JSON_ROW, dtype=np.uint8)
    row[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return row


def read_robot_state(robot):
    joints = robot.current_joint_state
    pose = robot.current_cartesian_state.pose.end_effector_pose
    elbow = robot.state.elbow
    flip = elbow.joint_4_flip
    return {
        "q": np.asarray(joints.position, dtype=np.float64).reshape(7),
        "dq": np.asarray(joints.velocity, dtype=np.float64).reshape(7),
        "O_T_EE": np.asarray(pose.matrix, dtype=np.float64).reshape(4, 4),
        "translation": np.asarray(pose.translation, dtype=np.float64).reshape(3),
        "quaternion_xyzw": np.asarray(pose.quaternion, dtype=np.float64).reshape(4),
        "elbow": np.array([float(elbow.joint_3_pos), np.nan if flip is None else float(int(flip))]),
    }


def sample(robot, listener, gripper, runner, t0):
    now = time.monotonic()
    record = read_robot_state(robot)
    record["t"] = now - t0
    record["wall_time"] = time.time()
    record["phase"] = runner.phase
    record["target_q"] = runner.positions[runner.phase]
    record["gripper_width"] = gripper.width if gripper else float("nan")
    packets = listener.snapshot()
    depth = np.full((len(SENSORS), 2 * ZONES * ZONES), NO_TARGET_MM, dtype=np.uint16)
    age = np.full(len(SENSORS), np.nan, dtype=np.float32)
    count = np.zeros(len(SENSORS), dtype=np.int64)
    for index, key in enumerate(SENSORS):
        if key in packets:
            depth[index], arrived, count[index] = packets[key]
            age[index] = (now - arrived) * 1000.0
    record["depth_mm"], record["age_ms"], record["count"] = depth, age, count
    return record


def save_h5(path, records, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    T = len(records)
    stack = lambda key: np.stack([record[key] for record in records])
    q, dq = stack("q"), stack("dq")
    width = stack("gripper_width")
    finger = np.where(np.isfinite(width), width / 2.0, 0.0)
    finger_v = np.gradient(finger, stack("t")) if T > 1 else np.zeros(T)
    depth, age, count = stack("depth_mm"), stack("age_ms"), stack("count")
    fresh = np.concatenate([count[:1] > 0, np.diff(count, axis=0) > 0])
    translation, quat_xyzw = stack("translation"), stack("quaternion_xyzw")

    with h5py.File(path, "w") as handle:
        handle.attrs["sim"] = False
        traj = handle.create_group("traj_0")
        for key, value in metadata.items():
            traj.attrs[key] = value

        agent = traj.create_group("obs/agent")
        agent.create_dataset("qpos", data=np.stack([
            json_row({"arm": q[t].tolist(), "base": [], "gripper": [float(finger[t])] * 2}) for t in range(T)]))
        agent.create_dataset("qvel", data=np.stack([
            json_row({"arm": dq[t].tolist(), "base": [], "gripper": [float(finger_v[t])] * 2}) for t in range(T)]))

        extra = traj.create_group("obs/extra")
        tcp = np.concatenate([translation, quat_xyzw[:, [3, 0, 1, 2]]], axis=1)
        extra.create_dataset("tcp_pose", data=tcp.astype(np.float32))
        extra["tcp_pose"].attrs["layout"] = "x, y, z, qw, qx, qy, qz of O_T_EE in the base frame"
        extra.create_dataset("robot_base_pose", data=np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (T, 1)))
        extra.create_dataset("policy_phase", data=stack("phase").astype(np.int64))
        extra["policy_phase"].attrs["meaning"] = "index of the waypoint being executed"

        proximity = traj.create_group("obs/proximity")
        proximity.attrs["unit"] = "m"
        proximity.attrs["no_target_m"] = NO_TARGET_M
        proximity.attrs["source"] = "block 1 (confident zones) of the newest packet at each tick"
        block1 = depth[:, :, : ZONES * ZONES].astype(np.float32) / 1000.0
        block1[(block1 <= 0) | (block1 >= NO_TARGET_M)] = NO_TARGET_M
        for index, (name, sensor_id) in enumerate(SENSORS):
            proximity.create_dataset(sim_name(name, sensor_id),
                                     data=block1[:, index].reshape(T, 1, ZONES, ZONES))

        targets = stack("target_q")
        actions = traj.create_group("actions")
        actions.create_dataset("joint_pos", data=np.stack([
            json_row({"arm": targets[t].tolist(), "gripper": [0.0]}) for t in range(T)]))
        actions.create_dataset("commanded_action", data=np.stack([
            json_row({"arm": targets[t].tolist(), "base": [], "gripper": [0.0]}) for t in range(T)]))
        actions.attrs["meaning"] = "joint target of the current waypoint (franky JointMotion)"

        for key in ("success", "fail", "truncated"):
            traj.create_dataset(key, data=np.zeros(T, dtype=bool))
        terminated = np.zeros(T, dtype=bool)
        terminated[-1] = True
        traj.create_dataset("terminated", data=terminated)
        traj.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))

        real = traj.create_group("real")
        real.create_dataset("t", data=stack("t"))
        real["t"].attrs["unit"] = "s since the first tick (monotonic clock)"
        real.create_dataset("wall_time", data=stack("wall_time"))
        real.create_dataset("q", data=q)
        real.create_dataset("dq", data=dq)
        real.create_dataset("O_T_EE", data=stack("O_T_EE"))
        real["O_T_EE"].attrs["layout"] = "4x4 homogeneous, translation in the last column"
        real.create_dataset("elbow", data=stack("elbow"))
        real.create_dataset("target_q", data=targets)
        real.create_dataset("gripper_width", data=width)
        real.create_dataset("sensor_names", data=np.array(
            [sim_name(name, sensor_id) for name, sensor_id in SENSORS], dtype=h5py.string_dtype()))
        real.create_dataset("sensor_link", data=np.array([name for name, _ in SENSORS], dtype=h5py.string_dtype()))
        real.create_dataset("sensor_id", data=np.array([sensor_id for _, sensor_id in SENSORS], dtype=np.uint8))
        real.create_dataset("depth_mm", data=depth.reshape(T, len(SENSORS), 2, ZONES, ZONES), compression="gzip")
        real["depth_mm"].attrs["layout"] = "(tick, sensor, block, row, col); block 0 confident, block 1 with fills"
        real.create_dataset("age_ms", data=age)
        real["age_ms"].attrs["meaning"] = "time since the packet arrived; NaN = sensor never reported"
        real.create_dataset("fresh", data=fresh)
        real["fresh"].attrs["meaning"] = "a new packet arrived since the previous tick"
    return path


def collect(host, dynamics, pause, warmup_timeout, tail, use_gripper, allow_missing, output):
    from franky import Robot

    sockets = open_sensor_sockets()
    listener = SensorListener(sockets)
    listener.start()
    gripper = None
    records = []
    try:
        deadline = time.monotonic() + warmup_timeout
        while listener.missing() and time.monotonic() < deadline:
            time.sleep(0.05)
        missing = listener.missing()
        if missing:
            message = "no packets from: " + ", ".join(missing)
            if not allow_missing:
                raise TimeoutError(message + " (pass --allow-missing to record anyway)")
            print("warning: " + message)

        robot = Robot(host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = dynamics
        if use_gripper:
            try:
                gripper = GripperReader(host)
                gripper.start()
            except Exception as exc:
                print(f"warning: no gripper width ({exc})")
                gripper = None

        runner = MotionRunner(robot, POSITIONS, pause)
        period = 1.0 / RATE_HZ
        t0 = time.monotonic()
        runner.start()
        tick = 0
        stop_at = None
        while True:
            records.append(sample(robot, listener, gripper, runner, t0))
            if runner.done.is_set() and stop_at is None:
                stop_at = time.monotonic() + tail
            if stop_at is not None and time.monotonic() >= stop_at:
                break
            tick += 1
            sleep = t0 + tick * period - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                tick = int((time.monotonic() - t0) / period) + 1     # fell behind: skip, keep the grid
        if runner.error is not None:
            print(f"motion stopped early: {runner.error}")
    finally:
        listener.stop_event.set()
        listener.join(timeout=1.0)
        if gripper:
            gripper.stop_event.set()
        for sock in sockets:
            sock.close()
        if records:
            metadata = {
                "host": host,
                "relative_dynamics_factor": dynamics,
                "multicast_group": GROUP,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "policy_dt_ms": 1000.0 / RATE_HZ,
                "rate_hz": RATE_HZ,
                "waypoints": POSITIONS,
                "pause_s": pause,
                "bad_packets": listener.bad,
            }
            saved = save_h5(output, records, metadata)
            report(records, saved)


def report(records, path):
    t = np.array([record["t"] for record in records])
    count = np.stack([record["count"] for record in records])
    age = np.stack([record["age_ms"] for record in records])
    dt = np.diff(t) * 1000.0 if len(t) > 1 else np.zeros(1)
    span = max(t[-1] - t[0], 1e-9)
    rates = (count[-1] - count[0]) / span
    print(f"saved {len(records)} ticks over {span:.1f} s to {path}")
    print(f"tick period {dt.mean():.1f} ms mean, {dt.max():.1f} ms max (target {1000 / RATE_HZ:.1f})")
    print(f"sensor packet rate {np.nanmin(rates):.1f}-{np.nanmax(rates):.1f} Hz, "
          f"packet age median {np.nanmedian(age):.0f} ms, max {np.nanmax(age):.0f} ms")
    slow = [f"{sim_name(*SENSORS[i])} {rates[i]:.1f} Hz" for i in np.flatnonzero(rates < RATE_HZ * 0.5)]
    if slow:
        print("sensors under half the tick rate: " + ", ".join(slow))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument("--dynamics", type=float, default=0.05, help="relative velocity, acceleration, and jerk")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds to hold at each waypoint (still logging)")
    parser.add_argument("--tail", type=float, default=0.5, help="seconds to keep logging after the last waypoint")
    parser.add_argument("--warmup-timeout", type=float, default=5.0,
                        help="seconds to wait for every sensor to report before moving")
    parser.add_argument("--allow-missing", action="store_true", help="record even if some sensors never report")
    parser.add_argument("--no-gripper", action="store_true", help="do not read the Franka Hand width")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HDF5 output path (default: recordings/<utc timestamp>.h5)",
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("recordings") / f"{stamp}.h5"
    collect(args.host, args.dynamics, args.pause, args.warmup_timeout, args.tail,
            not args.no_gripper, args.allow_missing, output)


if __name__ == "__main__":
    main()
