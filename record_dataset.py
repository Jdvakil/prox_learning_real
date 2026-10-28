"""Move the Franka through joint waypoints and log the robot, all 41 proximity sensors and both D435s.

The arm runs in a background thread while the main loop samples on a fixed 15 Hz clock, so the
log covers the motion itself, not only the stops. A listener thread keeps the newest packet of
every sensor; each tick stores that packet and how old it is. Each camera (wrist, exocentric) is
recorded at its full rate (30 fps): every frame is streamed to disk while the arm moves, and the
newest frame at each 15 Hz tick also goes into a per-tick video for the ACT converter.

One recording is a row folder:

  recordings/<utc stamp>/
      trajectory.h5                       layout below
      episode_00000000_wrist_camera.mp4   RGB, one frame per 15 Hz tick (convert_pact_place_to_act)
      episode_00000000_exo_camera_1.mp4
      wrist_camera_30fps.mp4              RGB, every camera frame
      exo_camera_1_30fps.mp4

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
  traj_0/real/cameras/<name>/      every camera frame (N at 30 fps): depth_mm (N, H, W) uint16 mm
                                   aligned to colour, t, frame_number, device_time_ms; rgb (N, H, W, 3)
                                   with --rgb-h5; tick_frame (T,) = index of the frame used at each tick;
                                   intrinsics and serial as attributes

<sim name> is the camera in assets/robots/franka_skin/model_hybrid_franka_hand.xml
(sensor_layout.sim_name). Each packet is 1 id byte + 128 uint16 mm: two row-major 8x8 blocks,
block 1 confident zones only (4000 = no target), block 2 the same plus low-confidence fills.

Requires franky (built from submodules/franky), h5py, pyrealsense2 and imageio[ffmpeg] (or opencv).
"""

import argparse
import json
import queue
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


class Mp4Writer:
    """Streaming RGB uint8 -> H.264 mp4 (imageio-ffmpeg), or MPEG-4 through OpenCV without imageio."""

    def __init__(self, path, fps):
        self.path, self.fps = str(path), fps
        self.writer = None
        self.cv2 = None

    def append(self, frame):
        if self.writer is None:
            try:
                import imageio.v2 as imageio

                self.writer = imageio.get_writer(self.path, fps=self.fps, codec="libx264", quality=8,
                                                 macro_block_size=1, ffmpeg_log_level="error")
            except ImportError:
                import cv2

                self.cv2 = cv2
                height, width = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if self.cv2 is None:
            self.writer.append_data(frame)
        else:
            self.writer.write(self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is None:
            return
        if self.cv2 is None:
            self.writer.close()
        else:
            self.writer.release()


class CameraRecorder(threading.Thread):
    """Writes every frame of one RealSenseCamera to disk while recording.

    The camera thread hands each frame over through cam.sink; this thread appends the depth to a
    resizable dataset in the open trajectory.h5 (and the RGB with rgb_h5) and the RGB to
    <name>_30fps.mp4. tick() queues the RGB used at a 15 Hz tick for the per-tick episode mp4.
    """

    def __init__(self, cam, group, row, store_depth, store_rgb):
        super().__init__(daemon=True)
        self.cam, self.group = cam, group
        self.store_depth, self.store_rgb = store_depth, store_rgb
        self.queue = queue.Queue()
        self.full = Mp4Writer(row / f"{cam.name}_30fps.mp4", cam.fps)
        self.episode = Mp4Writer(row / f"episode_00000000_{cam.name}.mp4", RATE_HZ)
        self.number, self.device_ms, self.arrived = [], [], []
        self.ticks = 0
        self.backlog = 0
        self.error = None
        self.depth = self.rgb = None

    def begin(self):
        self.start()
        self.cam.sink = lambda frame: self.queue.put(("frame", frame))

    def tick(self, rgb):
        if self.error is None:
            self.queue.put(("tick", rgb))

    def finish(self):
        self.cam.sink = None
        self.queue.put(("stop", None))
        self.join()
        if self.error is not None:
            print(f"warning: {self.cam.name} recorder stopped early: {self.error!r}")

    def write_frame(self, frame):
        rgb, depth, number, device_ms, arrived = frame
        n = len(self.number)
        height, width = depth.shape
        if self.store_depth:
            if self.depth is None:
                self.depth = self.group.create_dataset(
                    "depth_mm", shape=(0, height, width), maxshape=(None, height, width), dtype=np.uint16,
                    chunks=(1, height, width), compression="gzip", compression_opts=1, shuffle=True)
                self.depth.attrs["meaning"] = "z depth aligned to the colour image, mm, 0 = no depth"
            self.depth.resize(n + 1, axis=0)
            self.depth[n] = depth
        if self.store_rgb:
            if self.rgb is None:
                self.rgb = self.group.create_dataset(
                    "rgb", shape=(0, height, width, 3), maxshape=(None, height, width, 3), dtype=np.uint8,
                    chunks=(1, height, width, 3), compression="lzf")
            self.rgb.resize(n + 1, axis=0)
            self.rgb[n] = rgb
        self.full.append(rgb)
        self.number.append(number)
        self.device_ms.append(device_ms)
        self.arrived.append(arrived)

    def run(self):
        try:
            while True:
                self.backlog = max(self.backlog, self.queue.qsize())
                kind, item = self.queue.get()
                if kind == "stop":
                    break
                if kind == "frame":
                    self.write_frame(item)
                else:
                    self.episode.append(item)
                    self.ticks += 1
        except Exception as exc:
            self.error = exc
            self.cam.sink = None
        finally:
            self.full.close()
            self.episode.close()

    def write_index(self, t0, tick_numbers):
        """Per-frame timestamps and the frame each 15 Hz tick used; call after finish()."""
        group, cam = self.group, self.cam
        for key, value in cam.intrinsics.items():
            group.attrs[key] = value
        group.attrs["serial"] = cam.serial
        group.attrs["fps"] = cam.fps
        group.attrs["depth_scale_m"] = 0.001
        number = np.array(self.number, dtype=np.int64)
        group.create_dataset("frame_number", data=number)
        group.create_dataset("device_time_ms", data=np.array(self.device_ms, dtype=np.float64))
        group.create_dataset("t", data=np.array(self.arrived, dtype=np.float64) - t0)
        group["t"].attrs["unit"] = "s since the first tick (monotonic clock), at frame arrival"
        tick_frame = np.searchsorted(number, np.asarray(tick_numbers, dtype=np.int64), side="right") - 1
        group.create_dataset("tick_frame", data=np.maximum(tick_frame, 0))
        group["tick_frame"].attrs["meaning"] = "index into the per-frame arrays of the newest frame at each tick"


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


def sensor_arrays(listener, now):
    """Newest packet of every sensor in SENSORS order: depth (S, 128) uint16, age ms, packet count."""
    packets = listener.snapshot()
    depth = np.full((len(SENSORS), 2 * ZONES * ZONES), NO_TARGET_MM, dtype=np.uint16)
    age = np.full(len(SENSORS), np.nan, dtype=np.float32)
    count = np.zeros(len(SENSORS), dtype=np.int64)
    for index, key in enumerate(SENSORS):
        if key in packets:
            depth[index], arrived, count[index] = packets[key]
            age[index] = (now - arrived) * 1000.0
    return depth, age, count


def wait_for_sensors(listener, timeout, allow_missing):
    deadline = time.monotonic() + timeout
    while listener.missing() and time.monotonic() < deadline:
        time.sleep(0.05)
    missing = listener.missing()
    if missing:
        message = "no packets from: " + ", ".join(missing)
        if not allow_missing:
            raise TimeoutError(message + " (pass --allow-missing to record anyway)")
        print("warning: " + message)


def sample(robot, listener, gripper, runner, recorders, t0):
    now = time.monotonic()
    record = read_robot_state(robot)
    record["t"] = now - t0
    record["wall_time"] = time.time()
    record["phase"] = runner.phase
    record["target_q"] = runner.positions[runner.phase]
    record["gripper_width"] = gripper.width if gripper else float("nan")
    record["depth_mm"], record["age_ms"], record["count"] = sensor_arrays(listener, now)
    for rec in recorders:
        (rgb, _, number, _, arrived), _ = rec.cam.latest()
        rec.tick(rgb)
        record[rec.cam.name] = (number, (now - arrived) * 1000.0)
    return record


def write_sensors(traj, depth, age, count):
    """obs/proximity/<sim name> (block 1 in m) and the raw packets under real/, for (T, S, 128) depth."""
    T = len(depth)
    proximity = traj.create_group("obs/proximity")
    proximity.attrs["unit"] = "m"
    proximity.attrs["no_target_m"] = NO_TARGET_M
    proximity.attrs["source"] = "block 1 (confident zones) of the newest packet at each tick"
    block1 = depth[:, :, : ZONES * ZONES].astype(np.float32) / 1000.0
    block1[(block1 <= 0) | (block1 >= NO_TARGET_M)] = NO_TARGET_M
    for index, (name, sensor_id) in enumerate(SENSORS):
        proximity.create_dataset(sim_name(name, sensor_id),
                                 data=block1[:, index].reshape(T, 1, ZONES, ZONES))

    real = traj.require_group("real")
    fresh = np.concatenate([count[:1] > 0, np.diff(count, axis=0) > 0])
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


def save_h5(handle, records, metadata):
    """Write the 15 Hz ticks into an open trajectory.h5 (camera frames are already in it)."""
    T = len(records)
    stack = lambda key: np.stack([record[key] for record in records])
    q, dq = stack("q"), stack("dq")
    width = stack("gripper_width")
    finger = np.where(np.isfinite(width), width / 2.0, 0.0)
    finger_v = np.gradient(finger, stack("t")) if T > 1 else np.zeros(T)
    translation, quat_xyzw = stack("translation"), stack("quaternion_xyzw")

    handle.attrs["sim"] = False
    traj = handle.require_group("traj_0")
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

    write_sensors(traj, stack("depth_mm"), stack("age_ms"), stack("count"))

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

    real = traj.require_group("real")
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


def open_cameras(args):
    """Start the wrist (required) and exocentric RealSense threads and wait for auto exposure."""
    from cameras import RealSenseCamera

    if not args.wrist_serial:
        raise SystemExit("--wrist-serial is required (python cameras.py lists the cameras), or pass --no-cameras")
    cameras = []
    try:
        for name, serial in (("wrist_camera", args.wrist_serial), ("exo_camera_1", args.exo_serial)):
            if serial:
                cameras.append(RealSenseCamera(name, serial, args.cam_width, args.cam_height, args.cam_fps))
        for cam in cameras:
            cam.start()
        for cam in cameras:
            cam.wait_ready()
    except BaseException:
        for cam in cameras:
            cam.close()
        raise
    return cameras


def collect(args, row):
    from franky import Robot

    sockets = open_sensor_sockets()
    listener = SensorListener(sockets)
    listener.start()
    gripper = None
    cameras, recorders = [], []
    handle = None
    records = []
    t0 = time.monotonic()
    try:
        wait_for_sensors(listener, args.warmup_timeout, args.allow_missing)
        if not args.no_cameras:
            cameras = open_cameras(args)
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = args.dynamics
        if not args.no_gripper:
            try:
                gripper = GripperReader(args.host)
                gripper.start()
            except Exception as exc:
                print(f"warning: no gripper width ({exc})")
                gripper = None

        row.mkdir(parents=True, exist_ok=False)
        handle = h5py.File(row / "trajectory.h5", "w")
        for cam in cameras:
            group = handle.create_group(f"traj_0/real/cameras/{cam.name}")
            recorders.append(CameraRecorder(cam, group, row, not args.no_depth, args.rgb_h5))

        runner = MotionRunner(robot, POSITIONS, args.pause)
        period = 1.0 / RATE_HZ
        for rec in recorders:
            rec.begin()
        t0 = time.monotonic()
        runner.start()
        tick = 0
        stop_at = None
        while True:
            records.append(sample(robot, listener, gripper, runner, recorders, t0))
            if runner.done.is_set() and stop_at is None:
                stop_at = time.monotonic() + args.tail
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
        for rec in recorders:
            rec.finish()
        listener.stop_event.set()
        listener.join(timeout=1.0)
        if gripper:
            gripper.stop_event.set()
        for sock in sockets:
            sock.close()
        for cam in cameras:
            cam.close()
        if handle is not None:
            if records:
                for rec in recorders:
                    rec.write_index(t0, [record[rec.cam.name][0] for record in records])
                metadata = {
                    "host": args.host,
                    "relative_dynamics_factor": args.dynamics,
                    "multicast_group": GROUP,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "policy_dt_ms": 1000.0 / RATE_HZ,
                    "rate_hz": RATE_HZ,
                    "waypoints": POSITIONS,
                    "pause_s": args.pause,
                    "bad_packets": listener.bad,
                    "cameras": [cam.name for cam in cameras],
                }
                save_h5(handle, records, metadata)
            handle.close()
            if records:
                report(records, recorders, row)


def report(records, recorders, path):
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
    for rec in recorders:
        number = np.array(rec.number)
        if len(number) < 2:
            print(f"{rec.cam.name}: {len(number)} frames")
            continue
        arrived = np.array(rec.arrived)
        fps = (len(number) - 1) / max(arrived[-1] - arrived[0], 1e-9)
        dropped = int(np.sum(np.diff(number) - 1))
        ages = np.array([record[rec.cam.name][1] for record in records])
        print(f"{rec.cam.name}: {len(number)} frames at {fps:.1f} fps (target {rec.cam.fps}), "
              f"{dropped} dropped by the camera, write backlog max {rec.backlog}; "
              f"{rec.ticks} tick frames, age median {np.median(ages):.0f} ms")


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
    parser.add_argument("--wrist-serial", default=None, help="serial of the wrist D435 (python cameras.py)")
    parser.add_argument("--exo-serial", default=None, help="serial of the exocentric D435")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30, help="camera rate; every frame is recorded")
    parser.add_argument("--no-depth", action="store_true", help="do not store depth (~9 MB per second per camera)")
    parser.add_argument("--rgb-h5", action="store_true", help="also store raw RGB of every frame in trajectory.h5")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="row folder to create (default: recordings/<utc timestamp>/)",
    )
    args = parser.parse_args()
    row = args.output
    if row is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        row = Path("recordings") / stamp
    collect(args, row)


if __name__ == "__main__":
    main()
