"""Kinesthetic teaching on the Franka, then replay with full logging (drawer pick-and-place data).

teach   The arm runs franky's joint impedance tracker with zero stiffness and zero damping: it
        floats under the robot's own gravity compensation and you move it by hand. The joint
        state is logged at 100 Hz. Keys, in this terminal:

            space  start / stop a recording     g  gripper: grasp / open
            h      go to the home pose          d  drop the recording in progress
            q      quit

        Each recording is saved as demos/<utc stamp>.h5: t, q, dq, O_T_EE, the commanded gripper
        state and the gripper width per sample, and the gripper events (time, sample, close/open).

replay  For each demo: move to its first pose (JointMotion), set the gripper, wait while you reset
        the scene, then stream the recorded joint path (q, dq) to a stiff joint impedance tracker
        at 100 Hz. At each recorded gripper event the path holds until the arm has settled and the
        gripper command has finished. Meanwhile a 15 Hz loop logs the 41 proximity sensors, the
        robot state, the commands, and RGB-D from the wrist and exocentric D435s. Each replay is
        one row, in the layout prox_learning's scripts/convert_pact_place_to_act.py converts for
        ACT / PC-ACT:

            dataset/rows/<utc stamp>_<demo>/
                trajectory.h5                       sim datagen layout (traj_0/...) + traj_0/real/
                episode_00000000_wrist_camera.mp4   RGB, one frame per 15 Hz tick
                episode_00000000_exo_camera_1.mp4
                result.json                         {"task_success": ...} from your answer

trajectory.h5, T ticks at 15 Hz:

  traj_0/obs/agent/qpos, qvel       (T, 2000) uint8 JSON {"arm": [7], "base": [], "gripper": [2]},
                                    gripper = finger half widths
  traj_0/obs/extra/tcp_pose         (T, 7) [x, y, z, qw, qx, qy, qz] of O_T_EE
  traj_0/obs/proximity/<sim name>   (T, 1, 8, 8) m, block 1 of the newest packet, 4.0 = no target
  traj_0/actions/joint_pos          (T, 2000) uint8 JSON {"arm": commanded q, "gripper": [0 | 255]},
                                    0 = open, 255 = closed, as in the sim datasets
  traj_0/real/                      q, dq, q_ref, dq_ref, tau_J, tau_J_d, O_T_EE, elbow, gripper
                                    width and command, both sensor blocks in mm with packet age,
                                    timestamps; cameras/<name>/depth_mm (T, H, W) uint16 with frame
                                    numbers, ages and intrinsics
  traj_0/teach/                     the demo that was replayed

Needs franky (submodules/franky), h5py, numpy, pyrealsense2 and imageio[ffmpeg] (or opencv).
"""

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from record_dataset import (
    GROUP,
    RATE_HZ,
    SensorListener,
    json_row,
    open_sensor_sockets,
    sensor_arrays,
    wait_for_sensors,
    write_sensors,
)

TEACH_HZ = 100.0
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
# FR3 joint limits [rad]. While teaching, franky pushes back softly within 0.1 rad of them.
Q_MIN = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
Q_MAX = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
# Replay gains, from libfranka's joint impedance example.
REPLAY_STIFFNESS = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
REPLAY_DAMPING = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
# --friction-comp values, from franky's zero-g example.
FRICTION_COULOMB = [0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2]
FRICTION_VISCOUS = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
DEFAULT_TORQUE_THRESHOLD = 20.0   # franky's defaults, restored after teaching
DEFAULT_FORCE_THRESHOLD = 30.0
GRIP_OPEN, GRIP_CLOSED = 0.0, 255.0
SETTLE_TOL = 0.01                 # rad, arm settled at a gripper event
SETTLE_TIMEOUT = 1.5
GRIP_TIMEOUT = 5.0

KEYS_HELP = """zero-gain mode: move the arm by hand
  space  start / stop a recording     g  gripper: grasp / open
  h      go to the home pose          d  drop the recording in progress
  q      quit"""


class Console:
    """Single keys from this terminal (cbreak mode) plus line prompts.

    `kind` names each prompt (home, save, recover, reset, result) so a scripted stand-in can
    answer it; see sim/twin.py.
    """

    def __enter__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        if self.fd is not None:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.fd = None
        return False

    def keys(self):
        if getattr(self, "fd", None) is None:
            return []
        out = []
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            out.extend(chunk.decode(errors="ignore"))
        return out

    def ask(self, kind, question):
        cbreak = getattr(self, "fd", None) is not None
        if cbreak:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        try:
            return input(question).strip().lower()
        finally:
            if cbreak:
                tty.setcbreak(self.fd)


class GripperWorker(threading.Thread):
    """The Franka Hand: width read continuously, grasp / open started without blocking."""

    def __init__(self, host, speed, force):
        super().__init__(daemon=True)
        from franky import Gripper

        self.gripper = Gripper(host)
        self.speed, self.force = speed, force
        try:
            self.width = float(self.gripper.width)
        except Exception:
            self.width = float("nan")
        self.closed = False          # commanded state
        self.pending = None          # future of the running command
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.width = float(self.gripper.width)
            except Exception:
                self.width = float("nan")
                time.sleep(0.05)
            time.sleep(0.01)

    def command(self, close):
        self.closed = bool(close)
        if close:   # unknown object width: any width counts as grasped
            self.pending = self.gripper.grasp_async(0.0, self.speed, self.force, epsilon_inner=0.005,
                                                    epsilon_outer=1.0)
        else:
            self.pending = self.gripper.open_async(self.speed)

    def busy(self):
        return self.pending is not None and not self.pending.wait(0)

    def wait(self, timeout=GRIP_TIMEOUT):
        """True once the command finished and succeeded; False if it failed or timed out."""
        if self.pending is None:
            return True
        if not self.pending.wait(timeout):
            return False
        try:
            return bool(self.pending.get())
        except Exception as exc:
            print(f"gripper: {exc}")
            return False


def read_state(robot):
    """One read of the arm state."""
    s = robot.state
    flip = s.elbow.joint_4_flip
    return {
        "q": np.asarray(s.q, dtype=np.float64).reshape(7),
        "dq": np.asarray(s.dq, dtype=np.float64).reshape(7),
        "tau_J": np.asarray(s.tau_J, dtype=np.float64).reshape(7),
        "tau_J_d": np.asarray(s.tau_J_d, dtype=np.float64).reshape(7),
        "O_T_EE": np.asarray(s.O_T_EE.matrix, dtype=np.float64).reshape(4, 4),
        "elbow": np.array([float(s.elbow.joint_3_pos), np.nan if flip is None else float(int(flip))]),
    }


def quat_wxyz(R):
    """Unit quaternion [w, x, y, z] of a rotation matrix, w >= 0."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------------------- teach

def save_demo(path, samples, events, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, q, dq, pose, closed, width = (np.array(x) for x in zip(*samples))
    with h5py.File(path, "w") as f:
        for key, value in meta.items():
            f.attrs[key] = value
        f.create_dataset("t", data=t)
        f["t"].attrs["unit"] = "s since the recording started"
        f.create_dataset("q", data=q)
        f.create_dataset("dq", data=dq)
        f.create_dataset("O_T_EE", data=pose)
        f.create_dataset("gripper_closed", data=closed.astype(bool))
        f["gripper_closed"].attrs["meaning"] = "commanded gripper state (True = grasp)"
        f.create_dataset("gripper_width", data=width)
        ev = f.create_group("events")
        ev.create_dataset("t", data=np.array([e[0] for e in events], dtype=np.float64))
        ev.create_dataset("index", data=np.array([e[1] for e in events], dtype=np.int64))
        ev.create_dataset("closed", data=np.array([e[2] for e in events], dtype=bool))
        ev.attrs["meaning"] = "gripper key presses: time, last sample index, True = grasp / False = open"
    return path


def load_demo(path):
    with h5py.File(path, "r") as f:
        demo = {key: f[key][:] for key in ("t", "q", "dq", "O_T_EE", "gripper_closed", "gripper_width")}
        demo["event_t"] = f["events/t"][:]
        demo["event_index"] = f["events/index"][:]
        demo["event_closed"] = f["events/closed"][:]
        demo["attrs"] = dict(f.attrs)
    demo["start_closed"] = bool(demo["gripper_closed"][0])
    return demo


def teach(args, console):
    from franky import FrictionCompensationParams, JointImpedanceTracker, JointMotion, Robot

    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = args.dynamics
    gripper = GripperWorker(args.host, args.gripper_speed, args.grasp_force)
    if args.home_gripper:
        gripper.gripper.homing()
    gripper.start()
    gripper.command(False)
    gripper.wait()
    friction = None
    if args.friction_comp:
        friction = FrictionCompensationParams(coulomb=FRICTION_COULOMB, viscous=FRICTION_VISCOUS, max_torque=[2.0] * 7)

    def go_home():
        console.ask("home", "Enter: move to the home pose (keep clear of the arm) ")
        robot.set_collision_behavior(DEFAULT_TORQUE_THRESHOLD, DEFAULT_FORCE_THRESHOLD)
        robot.move(JointMotion(HOME))

    def float_arm():
        robot.set_collision_behavior(args.torque_threshold, args.force_threshold)
        return JointImpedanceTracker(robot, stiffness=[0.0] * 7, damping=[args.damping] * 7, friction=friction,
                                     lower_joint_limits=Q_MIN, upper_joint_limits=Q_MAX, period=1.0 / TEACH_HZ)

    def finish(samples, events, start_closed):
        if len(samples) < TEACH_HZ:
            print("recording shorter than 1 s, dropped")
            return
        if console.ask("save", f"{samples[-1][0]:.1f} s, {len(events)} gripper events. Save? [Y/n] ") == "n":
            print("dropped")
            return
        meta = {"host": args.host, "created_utc": datetime.now(timezone.utc).isoformat(), "rate_hz": TEACH_HZ,
                "home_q": HOME, "grasp_force": args.grasp_force, "gripper_speed": args.gripper_speed,
                "damping": args.damping, "friction_comp": args.friction_comp, "start_closed": start_closed}
        path = save_demo(args.out / f"{utc_stamp()}.h5", samples, events, meta)
        print(f"saved {path}")

    if not args.no_home:
        go_home()
    tracker = float_arm()
    print(KEYS_HELP)
    recording, samples, events, t_start, start_closed = False, [], [], 0.0, False
    try:
        with console:
            while True:
                if tracker.tick() is None:
                    try:
                        robot.join_motion()
                    except Exception as exc:
                        print(f"\nrobot stopped: {exc}")
                    if recording:
                        print("recording dropped")
                        recording = False
                    console.ask("recover", "Enter: recover and float the arm again ")
                    robot.recover_from_errors()
                    tracker = float_arm()
                    continue
                now = time.monotonic()
                if recording:
                    state = read_state(robot)
                    samples.append((now - t_start, state["q"], state["dq"], state["O_T_EE"],
                                    gripper.closed, gripper.width))
                for key in console.keys():
                    if key == " ":
                        if not recording:
                            recording, samples, events = True, [], []
                            t_start, start_closed = now, gripper.closed
                            print("recording")
                        else:
                            recording = False
                            print("stopped")
                            finish(samples, events, start_closed)
                    elif key == "g":
                        if gripper.busy():
                            print("gripper still moving")
                            continue
                        gripper.command(not gripper.closed)
                        print("grasp" if gripper.closed else "open")
                        if recording:
                            events.append((now - t_start, max(len(samples) - 1, 0), gripper.closed))
                    elif key == "h":
                        if recording:
                            print("stop the recording first")
                            continue
                        tracker.stop()
                        go_home()
                        tracker = float_arm()
                    elif key == "d" and recording:
                        recording = False
                        print("recording dropped")
                    elif key == "q":
                        if recording:
                            recording = False
                            finish(samples, events, start_closed)
                        return
    finally:
        tracker.stop()
        robot.set_collision_behavior(DEFAULT_TORQUE_THRESHOLD, DEFAULT_FORCE_THRESHOLD)
        gripper.stop_event.set()


# --------------------------------------------------------------------------------------- replay

class ReplayRunner(threading.Thread):
    """Streams one demo's joint path to the impedance tracker; holds at each gripper event."""

    def __init__(self, robot, gripper, demo, speed, stiffness, damping, max_error, tail):
        super().__init__(daemon=True)
        self.robot, self.gripper, self.demo = robot, gripper, demo
        self.speed, self.stiffness, self.damping = speed, stiffness, damping
        self.max_error, self.tail = max_error, tail
        self.q_ref, self.dq_ref = demo["q"][0].copy(), np.zeros(7)
        self.clock = 0.0             # demo time being replayed
        self.closed = demo["start_closed"]
        self.phase = 0               # gripper events done
        self.max_seen_error = 0.0
        self.grip_failures = 0
        self.error = None
        self.abort = threading.Event()
        self.done = threading.Event()

    def reference(self, clock, moving):
        t, q, dq = self.demo["t"], self.demo["q"], self.demo["dq"]
        q_ref = np.array([np.interp(clock, t, q[:, j]) for j in range(7)])
        if not moving:
            return q_ref, np.zeros(7)
        return q_ref, np.array([np.interp(clock, t, dq[:, j]) for j in range(7)]) * self.speed

    def run(self):
        from franky import JointImpedanceTracker

        events = list(zip(self.demo["event_t"], self.demo["event_closed"]))
        end = float(self.demo["t"][-1])
        try:
            with JointImpedanceTracker(self.robot, stiffness=self.stiffness, damping=self.damping,
                                       period=1.0 / TEACH_HZ) as tracker:
                step, since, nxt = "track", 0.0, 0
                while True:
                    dt = tracker.tick()
                    if dt is None:
                        raise RuntimeError("robot left control (reflex or error)")
                    if self.abort.is_set():
                        raise RuntimeError("aborted")
                    now = time.monotonic()
                    if step == "track":
                        self.clock = min(self.clock + dt * self.speed, end)
                        if nxt < len(events) and self.clock >= events[nxt][0]:
                            self.clock, step, since = float(events[nxt][0]), "settle", now
                        elif self.clock >= end:
                            step, since = "tail", now
                    self.q_ref, self.dq_ref = self.reference(self.clock, step == "track")
                    tracker.set_target(self.q_ref, self.dq_ref)
                    error = float(np.abs(np.asarray(self.robot.state.q) - self.q_ref).max())
                    self.max_seen_error = max(self.max_seen_error, error)
                    if error > self.max_error:
                        raise RuntimeError(f"tracking error {error:.3f} rad > --max-error {self.max_error}")
                    if step == "settle" and (error < SETTLE_TOL or now - since > SETTLE_TIMEOUT):
                        self.gripper.command(events[nxt][1])
                        self.closed = bool(events[nxt][1])
                        step, since = "grip", now
                    elif step == "grip" and not self.gripper.busy():
                        if not self.gripper.wait(0):
                            self.grip_failures += 1
                        nxt += 1
                        self.phase = nxt
                        step = "track"
                    elif step == "grip" and now - since > GRIP_TIMEOUT:
                        raise RuntimeError("gripper command timed out")
                    elif step == "tail" and now - since >= self.tail:
                        break
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()


def sample_tick(robot, listener, gripper, cameras, runner, t0):
    now = time.monotonic()
    record = read_state(robot)
    record["t"], record["wall_time"] = now - t0, time.time()
    record["q_ref"], record["dq_ref"], record["clock"] = runner.q_ref, runner.dq_ref, runner.clock
    record["closed"], record["phase"] = runner.closed, runner.phase
    record["gripper_width"] = gripper.width
    record["depth_mm"], record["age_ms"], record["count"] = sensor_arrays(listener, now)
    for cam in cameras:
        (rgb, depth, number, device_ms, arrived), count = cam.latest()
        record[cam.name] = (rgb, depth, number, device_ms, (now - arrived) * 1000.0, count)
    return record


def run_replay(robot, gripper, listener, cameras, demo, args):
    runner = ReplayRunner(robot, gripper, demo, args.speed, args.stiffness, args.damping, args.max_error, args.tail)
    period = 1.0 / RATE_HZ
    records = []
    t0 = time.monotonic()
    runner.start()
    tick = 0
    try:
        while True:
            records.append(sample_tick(robot, listener, gripper, cameras, runner, t0))
            if runner.done.is_set():
                break
            tick += 1
            sleep = t0 + tick * period - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                tick = int((time.monotonic() - t0) / period) + 1     # fell behind: skip, keep the grid
    except BaseException:          # Ctrl-C: let the runner bring the arm to rest before leaving
        runner.abort.set()
        runner.join(timeout=5.0)
        raise
    runner.join()
    return records, runner


def write_mp4(path, frames, fps):
    """RGB uint8 frames -> H.264 mp4 (imageio-ffmpeg), or MPEG-4 through OpenCV without imageio."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    if imageio is not None:
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_log_level="error") as writer:
            for frame in frames:
                writer.append_data(frame)
        return
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def save_row(row, records, cameras, demo, demo_path, meta, success, rgb_h5, depth_h5):
    row.mkdir(parents=True, exist_ok=False)
    T = len(records)
    stack = lambda key: np.stack([record[key] for record in records])
    q, dq, q_ref = stack("q"), stack("dq"), stack("q_ref")
    width = stack("gripper_width")
    finger = np.where(np.isfinite(width), width / 2.0, 0.0)
    finger_v = np.gradient(finger, stack("t")) if T > 1 else np.zeros(T)
    closed = stack("closed").astype(bool)
    grip = np.where(closed, GRIP_CLOSED, GRIP_OPEN)
    pose = stack("O_T_EE")

    with h5py.File(row / "trajectory.h5", "w") as f:
        f.attrs["sim"] = False
        traj = f.create_group("traj_0")
        for key, value in meta.items():
            traj.attrs[key] = value

        agent = traj.create_group("obs/agent")
        agent.create_dataset("qpos", data=np.stack([
            json_row({"arm": q[t].tolist(), "base": [], "gripper": [float(finger[t])] * 2}) for t in range(T)]))
        agent.create_dataset("qvel", data=np.stack([
            json_row({"arm": dq[t].tolist(), "base": [], "gripper": [float(finger_v[t])] * 2}) for t in range(T)]))

        extra = traj.create_group("obs/extra")
        tcp = np.concatenate([pose[:, :3, 3], np.stack([quat_wxyz(p[:3, :3]) for p in pose])], axis=1)
        extra.create_dataset("tcp_pose", data=tcp.astype(np.float32))
        extra["tcp_pose"].attrs["layout"] = "x, y, z, qw, qx, qy, qz of O_T_EE in the base frame"
        extra.create_dataset("robot_base_pose", data=np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (T, 1)))
        extra.create_dataset("policy_phase", data=stack("phase").astype(np.int64))
        extra["policy_phase"].attrs["meaning"] = "number of demo gripper events already replayed"

        write_sensors(traj, stack("depth_mm"), stack("age_ms"), stack("count"))

        actions = traj.create_group("actions")
        actions.create_dataset("joint_pos", data=np.stack([
            json_row({"arm": q_ref[t].tolist(), "gripper": [float(grip[t])]}) for t in range(T)]))
        actions.create_dataset("commanded_action", data=np.stack([
            json_row({"arm": q_ref[t].tolist(), "base": [], "gripper": [float(grip[t])]}) for t in range(T)]))
        actions.attrs["meaning"] = ("arm: joint reference streamed to the impedance tracker at the tick; "
                                    "gripper: 0 open / 255 grasp, the command in effect")

        done = np.zeros(T, dtype=bool)
        done[-1] = True
        traj.create_dataset("success", data=done & bool(success))
        traj.create_dataset("fail", data=done & (not success))
        traj.create_dataset("truncated", data=np.zeros(T, dtype=bool))
        traj.create_dataset("terminated", data=done)
        traj.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))

        real = traj.require_group("real")
        real.create_dataset("t", data=stack("t"))
        real["t"].attrs["unit"] = "s since the first tick (monotonic clock)"
        real.create_dataset("wall_time", data=stack("wall_time"))
        for key in ("q", "dq", "q_ref", "dq_ref", "tau_J", "tau_J_d", "O_T_EE", "elbow"):
            real.create_dataset(key, data=stack(key))
        real["O_T_EE"].attrs["layout"] = "4x4 homogeneous, translation in the last column"
        real.create_dataset("replay_time", data=stack("clock"))
        real["replay_time"].attrs["meaning"] = "demo time of the reference at the tick"
        real.create_dataset("target_q", data=q_ref)
        real.create_dataset("gripper_width", data=width)
        real.create_dataset("gripper_closed", data=closed)

        for cam in cameras:
            frames = [record[cam.name] for record in records]
            group = real.create_group(f"cameras/{cam.name}")
            for key, value in cam.intrinsics.items():
                group.attrs[key] = value
            group.attrs["serial"] = cam.serial
            group.attrs["fps"] = cam.fps
            group.attrs["depth_scale_m"] = 0.001
            height, width_px = frames[0][1].shape
            if depth_h5:                 # ~270 KB per D435 frame
                depth = group.create_dataset("depth_mm", shape=(T, height, width_px), dtype=np.uint16,
                                             chunks=(1, height, width_px), compression="gzip", compression_opts=1,
                                             shuffle=True)
                depth.attrs["meaning"] = "z depth aligned to the colour image, mm, 0 = no depth"
                for t, frame in enumerate(frames):
                    depth[t] = frame[1]
            if rgb_h5:
                rgb = group.create_dataset("rgb", shape=(T, height, width_px, 3), dtype=np.uint8,
                                           chunks=(1, height, width_px, 3), compression="lzf")
                for t, frame in enumerate(frames):
                    rgb[t] = frame[0]
            number = np.array([frame[2] for frame in frames], dtype=np.int64)
            group.create_dataset("frame_number", data=number)
            group.create_dataset("device_time_ms", data=np.array([frame[3] for frame in frames]))
            group.create_dataset("age_ms", data=np.array([frame[4] for frame in frames], dtype=np.float32))
            group["age_ms"].attrs["meaning"] = "time since the frame arrived"
            group.create_dataset("fresh", data=np.concatenate([[True], np.diff(number) != 0]))

        teach_group = traj.create_group("teach")
        teach_group.attrs["path"] = str(demo_path)
        for key in ("t", "q", "dq", "O_T_EE", "gripper_closed", "gripper_width", "event_t", "event_closed"):
            teach_group.create_dataset(key, data=demo[key])

    for cam in cameras:
        write_mp4(row / f"episode_00000000_{cam.name}.mp4", [record[cam.name][0] for record in records], RATE_HZ)
    return row


def report(records, runner, cameras, row):
    t = np.array([record["t"] for record in records])
    count = np.stack([record["count"] for record in records])
    age = np.stack([record["age_ms"] for record in records])
    dt = np.diff(t) * 1000.0 if len(t) > 1 else np.zeros(1)
    span = max(t[-1] - t[0], 1e-9)
    rates = (count[-1] - count[0]) / span
    print(f"{len(records)} ticks over {span:.1f} s, period {dt.mean():.1f} ms mean / {dt.max():.1f} ms max "
          f"(target {1000 / RATE_HZ:.1f})")
    print(f"sensors {np.nanmin(rates):.1f}-{np.nanmax(rates):.1f} Hz, packet age median {np.nanmedian(age):.0f} ms; "
          f"max tracking error {runner.max_seen_error * 1000:.1f} mrad; gripper events {runner.phase}"
          + (f" ({runner.grip_failures} failed)" if runner.grip_failures else ""))
    for cam in cameras:
        numbers = np.array([record[cam.name][2] for record in records])
        ages = np.array([record[cam.name][4] for record in records])
        print(f"{cam.name}: {len(np.unique(numbers))} distinct frames in {len(records)} ticks, "
              f"age median {np.median(ages):.0f} ms")
    if row is not None:
        print(f"saved {row}")


def replay(args, console):
    from franky import JointMotion, Robot

    demos = [(Path(p), load_demo(p)) for p in args.demos]
    sockets = open_sensor_sockets()
    listener = SensorListener(sockets)
    listener.start()
    cameras, gripper = [], None
    try:
        wait_for_sensors(listener, args.warmup_timeout, args.allow_missing)
        if not args.no_cameras:
            from cameras import start_cameras

            if not args.wrist_serial:
                raise SystemExit("--wrist-serial is required (python cameras.py lists the cameras), "
                                 "or pass --no-cameras")
            specs = [(name, serial) for name, serial in (("wrist_camera", args.wrist_serial),
                                                         ("exo_camera_1", args.exo_serial)) if serial]
            cameras = start_cameras(specs, args.cam_width, args.cam_height, args.cam_fps,
                                    reset=not args.no_camera_reset)

        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = args.dynamics
        robot.set_collision_behavior(args.torque_threshold, args.force_threshold)
        first = demos[0][1]["attrs"]
        gripper = GripperWorker(args.host, args.gripper_speed or float(first.get("gripper_speed", 0.05)),
                                args.grasp_force or float(first.get("grasp_force", 20.0)))
        if args.home_gripper:
            gripper.gripper.homing()
        gripper.start()
        rows = args.out / "rows"
        for demo_path, demo in demos:
            for rep in range(args.repeat):
                print(f"\n{demo_path.name} ({rep + 1}/{args.repeat}): {demo['t'][-1]:.1f} s, "
                      f"{len(demo['event_t'])} gripper events. Moving to its first pose.")
                robot.move(JointMotion(demo["q"][0].tolist()))
                gripper.command(demo["start_closed"])
                gripper.wait()
                answer = console.ask("reset", "Reset the scene as it was in the demo and step out of the camera "
                                              "views. Enter = replay, s = skip, q = quit ")
                if answer == "q":
                    return
                if answer == "s":
                    continue
                records, runner = run_replay(robot, gripper, listener, cameras, demo, args)
                if runner.error is not None:
                    print(f"replay stopped early: {runner.error}")
                report(records, runner, cameras, None)
                answer = console.ask("result", "Task done? y = success, n = failure, d = discard ")
                if runner.error is not None:
                    robot.recover_from_errors()
                if answer not in ("y", "n"):
                    print("discarded")
                    continue
                meta = {
                    "host": args.host, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "policy_dt_ms": 1000.0 / RATE_HZ, "rate_hz": RATE_HZ, "multicast_group": GROUP,
                    "demo": str(demo_path), "replay_index": rep, "speed": args.speed,
                    "stiffness": args.stiffness, "damping": args.damping, "bad_packets": listener.bad,
                    "cameras": [cam.name for cam in cameras], "task_success": answer == "y",
                    "replay_error": "" if runner.error is None else str(runner.error),
                }
                row = save_row(rows / f"{utc_stamp()}_{demo_path.stem}_r{rep}", records, cameras, demo, demo_path,
                               meta, answer == "y", args.rgb_h5, not args.no_depth)
                result = {"task_success": answer == "y", "demo": str(demo_path), "replay_index": rep,
                          "ticks": len(records), "created_utc": meta["created_utc"],
                          "replay_error": meta["replay_error"] or None}
                (row / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(f"saved {row}")
    finally:
        listener.stop_event.set()
        listener.join(timeout=1.0)
        for sock in sockets:
            sock.close()
        for cam in cameras:
            cam.close()
        if gripper:
            gripper.stop_event.set()


def main(argv=None, console=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    common.add_argument("--dynamics", type=float, default=0.1,
                        help="relative velocity / acceleration / jerk of the moves to home and to a demo start")
    common.add_argument("--grasp-force", type=float, default=None, help="N (teach default 20; replay: the demo's)")
    common.add_argument("--gripper-speed", type=float, default=None,
                        help="m/s (teach default 0.05; replay: the demo's)")
    common.add_argument("--home-gripper", action="store_true", help="home the Franka Hand first (once per power-up)")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("teach", parents=[common], help="record demos by moving the floating arm by hand")
    t.add_argument("--out", type=Path, default=Path("demos"))
    t.add_argument("--no-home", action="store_true", help="start floating where the arm is")
    t.add_argument("--damping", type=float, default=0.0, help="Nms/rad on every joint (0 = free)")
    t.add_argument("--friction-comp", action="store_true", help="add franky's Coulomb + viscous friction feedforward")
    t.add_argument("--torque-threshold", type=float, default=50.0, help="Nm collision reflex while teaching")
    t.add_argument("--force-threshold", type=float, default=80.0, help="N collision reflex while teaching")

    r = sub.add_parser("replay", parents=[common], help="replay demos and log sensors, robot and cameras")
    r.add_argument("demos", nargs="+", type=Path)
    r.add_argument("--out", type=Path, default=Path("dataset"))
    r.add_argument("--repeat", type=int, default=1, help="replays per demo")
    r.add_argument("--speed", type=float, default=1.0, help="replay speed relative to the demo")
    r.add_argument("--stiffness", type=float, nargs=7, default=REPLAY_STIFFNESS, help="Nm/rad")
    r.add_argument("--damping", type=float, nargs=7, default=REPLAY_DAMPING, help="Nms/rad")
    r.add_argument("--max-error", type=float, default=0.15, help="rad; stop if a joint lags the path by more")
    r.add_argument("--tail", type=float, default=1.0, help="s to hold and keep logging after the path ends")
    r.add_argument("--torque-threshold", type=float, default=DEFAULT_TORQUE_THRESHOLD)
    r.add_argument("--force-threshold", type=float, default=DEFAULT_FORCE_THRESHOLD)
    r.add_argument("--wrist-serial", default=None, help="serial of the wrist D435")
    r.add_argument("--exo-serial", default=None, help="serial of the exocentric D435")
    r.add_argument("--no-cameras", action="store_true")
    r.add_argument("--no-camera-reset", action="store_true",
                   help="skip the USB power-cycle before the cameras start (still retried once on failure)")
    r.add_argument("--cam-width", type=int, default=640)
    r.add_argument("--cam-height", type=int, default=480)
    r.add_argument("--cam-fps", type=int, default=30)
    r.add_argument("--rgb-h5", action="store_true", help="also store raw RGB in trajectory.h5 (large)")
    r.add_argument("--no-depth", action="store_true", help="do not store depth (saves ~8 MB per second)")
    r.add_argument("--warmup-timeout", type=float, default=5.0)
    r.add_argument("--allow-missing", action="store_true", help="replay even if some sensors never report")
    args = parser.parse_args(argv)
    console = console or Console()
    if args.command == "teach":
        args.grasp_force = 20.0 if args.grasp_force is None else args.grasp_force
        args.gripper_speed = 0.05 if args.gripper_speed is None else args.gripper_speed
        teach(args, console)
    else:
        if not 0.05 <= args.speed <= 1.5:
            raise SystemExit("--speed must be within 0.05-1.5")
        replay(args, console)


if __name__ == "__main__":
    main()
