"""Scripted block pick: go over a fixed block spot, close the gripper, lift, put the block back.

No model and no vision. The block has to sit where you showed the robot, every episode.

  1. Put the block at its spot. Move the arm by hand (Desk guiding mode) so the open fingers
     are around the block, then, with FCI active and the arm still:
         python data.py pick --set-pose                  # saves pick_pose.json (no motion)
  2. Check the trajectory without the robot:
         python data.py pick --plan-only
  3. Collect:
         python data.py pick --episodes 10 --wrist-serial <serial> --exo-serial <serial>
                                                         # -> recordings/pick_<utc stamp>/episode_0000/ ...

The whole episode is one smooth joint trajectory streamed at 100 Hz to franky's joint impedance
tracker (the same controller as data.py waypoints). Recorded: start at HOME with the gripper
open, a minimum-jerk joint move to --approach m above the grasp pose, a straight vertical
descent, close the gripper, a straight lift of --lift m, hold --hold s. Not recorded: lower the
block back onto its spot, open, rise, return HOME, so the next episode finds the block where it
was. Straight lines are solved with FR3 inverse kinematics from the joint angles saved by
--set-pose (the elbow stays as you posed it) and timed with a minimum-jerk profile, so every
segment starts and ends at rest. --jitter adds a random x/y offset (m) to the approach point;
the grasp pose never changes.

Rows use the record_dataset.py layout (trajectory.h5 + videos + result.json). Actions are the
joint reference sent to the tracker at each 15 Hz tick plus the gripper command (0 open,
1 close); the step is obs/extra/policy_phase (names in its "phases" attribute). An episode whose
gripper is not holding the block after the lift is saved with clean_success false and stops the
batch: put the block back and run again.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np

from . import live_dashboard
from . import record_dataset as base
from .kinesthetic import Q_MAX, Q_MIN, REPLAY_DAMPING, REPLAY_STIFFNESS

HOME = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]
POSE_FILE = Path(__file__).resolve().parents[1] / "pick_pose.json"
CONTROL_HZ = 100.0
START_TOLERANCE = 0.03      # rad; the tracker is not engaged unless the arm is this close to HOME
# franky ends a tracker with a damping-only TorqueStopMotion (zero stiffness, up to 2 s). The
# hand carries mass the robot's gravity model lacks, so the arm sinks ~0.1 rad/s meanwhile:
# 0.2 rad by default, which failed episode 2 on 2026-11-11. Stop fast, the robot then idles.
STOP_RAMP_S, STOP_MAX_S = 0.05, 0.3
PHASES = ["start", "approach", "descend", "grasp", "lift", "hold"]
GRIPPER_COMMAND = {"start": 0.0, "approach": 0.0, "descend": 0.0, "grasp": 1.0, "lift": 1.0, "hold": 1.0}

# FR3 kinematics (Franka documentation, Craig DH): a, d, alpha per joint, flange 0.107 m past joint 7.
DH_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
DH_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
DH_ALPHA = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]


def dh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def flange(q):
    out = np.eye(4)
    for i in range(7):
        out = out @ dh(DH_A[i], DH_D[i], DH_ALPHA[i], q[i])
    return out @ dh(0.0, 0.107, 0.0, 0.0)


def pose_error(target, current):
    """6-vector: position error (m) and rotation error as an axis-angle vector (rad)."""
    r = target[:3, :3] @ current[:3, :3].T
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    rot = axis * (0.5 if angle < 1e-6 else angle / (2 * np.sin(angle)))
    return np.concatenate([target[:3, 3] - current[:3, 3], rot])


def ik(target, seed, f_t_ee, iterations=100):
    """Damped least squares from seed; small steps keep the elbow where the seed has it."""
    q = np.array(seed, dtype=np.float64)
    for _ in range(iterations):
        current = flange(q) @ f_t_ee
        err = pose_error(target, current)
        if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-4:
            return q
        jac = np.empty((6, 7))
        for j in range(7):
            dq = np.zeros(7)
            dq[j] = 1e-6
            jac[:, j] = pose_error(flange(q + dq) @ f_t_ee, current) / 1e-6
        q = q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
    raise ValueError(f"IK did not converge (error {np.linalg.norm(err[:3]) * 1000:.2f} mm)")


def min_jerk(n):
    s = np.linspace(0, 1, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def samples(duration):
    return max(2, int(np.ceil(duration * CONTROL_HZ)) + 1)


def joint_segment(q0, q1, max_velocity):
    """Minimum-jerk joint move; peak speed 1.875 * distance / duration."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    duration = max(1.0, 1.875 * np.abs(q1 - q0).max() / max_velocity)
    return q0 + min_jerk(samples(duration))[:, None] * (q1 - q0)


def line_segment(q0, pose0, pose1, f_t_ee, speed):
    """Straight Cartesian line pose0 -> pose1 (same orientation), minimum-jerk timing, IK per tick."""
    duration = max(1.0, 1.875 * np.linalg.norm(pose1[:3, 3] - pose0[:3, 3]) / speed)
    path, q = [], np.asarray(q0, dtype=np.float64)
    for s in min_jerk(samples(duration)):
        target = pose1.copy()
        target[:3, 3] = pose0[:3, 3] + s * (pose1[:3, 3] - pose0[:3, 3])
        q = ik(target, q, f_t_ee)
        path.append(q)
    return np.array(path)


def hold_segment(q, duration):
    return np.repeat(np.asarray(q, dtype=np.float64)[None], samples(duration), axis=0)


def shifted(matrix, dx=0.0, dy=0.0, dz=0.0):
    out = np.array(matrix, dtype=np.float64)
    out[:3, 3] += [dx, dy, dz]
    return out


class Plan:
    """Every 100 Hz joint reference of one episode: recorded pick, then the unrecorded put-back."""

    def __init__(self, saved, args, rng):
        self.grasp = np.asarray(saved["grasp_pose"], dtype=np.float64)
        self.grasp_q = np.asarray(saved["grasp_q"], dtype=np.float64)
        self.f_t_ee = np.linalg.inv(flange(self.grasp_q)) @ self.grasp
        jitter = rng.uniform(-args.jitter, args.jitter, 2) if args.jitter > 0 else np.zeros(2)
        self.above = shifted(self.grasp, jitter[0], jitter[1], args.approach)
        self.lifted = shifted(self.grasp, dz=args.lift)
        above_q = ik(self.above, self.grasp_q, self.f_t_ee)
        descend = line_segment(above_q, self.above, self.grasp, self.f_t_ee, args.speed)
        lift = line_segment(descend[-1], self.grasp, self.lifted, self.f_t_ee, args.speed)
        self.pick = [("approach", joint_segment(HOME, above_q, args.max_velocity)),
                     ("descend", descend),
                     ("grasp", None),                                   # hold while the gripper closes
                     ("lift", lift),
                     ("hold", hold_segment(lift[-1], args.hold))]
        # Release --place-gap above the grasp height: a block that slipped in the fingers is
        # dropped a few mm, never pressed into the table and shoved sideways by a tilted hand.
        self.place = shifted(self.grasp, dz=args.place_gap)
        lower = line_segment(lift[-1], self.lifted, self.place, self.f_t_ee, args.speed)
        rise = line_segment(lower[-1], self.place, shifted(self.grasp, dz=args.approach), self.f_t_ee, args.speed)
        # A failed grasp still runs lift and hold, so the arm is at the lifted pose with nothing in
        # the hand: open, rise to the approach height, home. Never drop to the block again.
        rise_empty = line_segment(lift[-1], self.lifted, shifted(self.grasp, dz=max(args.approach, args.lift)),
                                  self.f_t_ee, args.speed)
        self.put_back = [("lower", lower), ("open", None), ("rise", rise),
                         ("home", joint_segment(rise[-1], HOME, args.max_velocity))]
        self.give_up = [("open", None), ("rise", rise_empty),
                        ("home", joint_segment(rise_empty[-1], HOME, args.max_velocity))]
        for sequence in (self.pick, self.pick + self.put_back, self.pick + self.give_up):
            last = np.asarray(HOME, dtype=np.float64)
            for name, path in sequence:
                if path is None:
                    continue
                if np.abs(path[0] - last).max() > 1e-6:     # every segment starts where the last ended
                    raise ValueError(f"{name}: starts {np.abs(path[0] - last).max():.4f} rad off the previous end")
                if np.any(path < np.asarray(Q_MIN) + 0.05) or np.any(path > np.asarray(Q_MAX) - 0.05):
                    raise ValueError(f"{name}: joint limit within 0.05 rad; move the block or re-pose")
                speed = np.abs(np.diff(path, axis=0)).max() * CONTROL_HZ if len(path) > 1 else 0.0
                if speed > 1.5 * args.max_velocity:
                    raise ValueError(f"{name}: joint speed {speed:.2f} rad/s (IK near a singularity?); re-pose")
                last = path[-1]

    def summary(self):
        lines = []
        for name, path in self.pick + self.put_back:
            if path is None:
                lines.append(f"  {name:<9} gripper")
                continue
            ee = flange(path[-1]) @ self.f_t_ee
            speed = np.abs(np.diff(path, axis=0)).max() * CONTROL_HZ if len(path) > 1 else 0.0
            lines.append(f"  {name:<9} {len(path) / CONTROL_HZ:5.2f} s  peak {speed:.2f} rad/s  "
                         f"ends at {np.round(ee[:3, 3], 3).tolist()} m")
        return "\n".join(lines)


class PickRunner(threading.Thread):
    """Streams the plan to the joint impedance tracker; the main loop logs at 15 Hz meanwhile."""

    def __init__(self, robot, gripper, plan, args, control_cpus=None):
        super().__init__(daemon=True)
        self.robot, self.gripper, self.plan, self.args, self.control_cpus = robot, gripper, plan, args, control_cpus
        self.lock = threading.Lock()
        self.phase = 0
        self.reference = np.asarray(HOME, dtype=np.float64)
        self.grasped = False
        self.max_seen_error = 0.0
        self.error = None
        self.abort = threading.Event()
        self.recorded = threading.Event()       # pick finished; the main loop stops recording
        self.resume = threading.Event()         # main loop finished writing; go put the block back
        self.done = threading.Event()

    def track(self, tracker, q, dq):
        if tracker.tick() is None:
            raise RuntimeError("robot left impedance control")
        with self.lock:                         # the logged action is always the reference in force
            tracker.set_target(q, dq)
            self.reference = q
        error = float(np.abs(np.asarray(self.robot.state.q) - q).max())
        self.max_seen_error = max(self.max_seen_error, error)
        if error > self.args.max_error:
            raise RuntimeError(f"tracking error {error:.3f} rad > {self.args.max_error}")

    def run_segments(self, tracker, segments, recorded):
        for name, path in segments:
            if self.abort.is_set():
                raise RuntimeError("aborted")
            if recorded:
                self.phase = PHASES.index(name)
            print(f"  {name}")
            if path is not None:
                dq = np.gradient(path, 1 / CONTROL_HZ, axis=0)
                for q, v in zip(path, dq):
                    self.track(tracker, q, v)
                continue
            # Gripper step: hold still at 100 Hz until the command finishes.
            if name == "grasp":   # unknown block width: any closed width counts (epsilon_outer = 1 m)
                pending = self.gripper.grasp_async(0.0, self.args.gripper_speed, self.args.grasp_force,
                                                   epsilon_inner=0.005, epsilon_outer=1.0)
            else:
                pending = self.gripper.open_async(self.args.gripper_speed)
            hold, zero, deadline = self.reference, np.zeros(7), time.monotonic() + 10.0
            while not pending.wait(0):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"gripper {name} did not finish in 10 s")
                self.track(tracker, hold, zero)
            try:
                ok = bool(pending.get())
            except Exception as exc:
                print(f"  gripper {name}: {exc}")
                ok = False
            if name == "grasp":       # closing on nothing also passes the epsilon test: check the width
                self.grasped = ok and float(self.gripper.width) > self.args.min_width

    def run(self):
        from franky import JointImpedanceTracker

        if self.control_cpus:
            os.sched_setaffinity(0, self.control_cpus)
        tracker = None
        try:
            offset = np.abs(np.asarray(self.robot.state.q) - HOME)
            if offset.max() > START_TOLERANCE:      # never engage the tracker with a jump in the target
                raise RuntimeError(f"arm is not at HOME (joint {int(offset.argmax()) + 1} off by "
                                   f"{offset.max():.3f} rad); not starting")
            # The tracker takes control when constructed and seeds its target with the current q.
            tracker = JointImpedanceTracker(self.robot, stiffness=REPLAY_STIFFNESS, damping=REPLAY_DAMPING,
                                            lower_joint_limits=Q_MIN, upper_joint_limits=Q_MAX,
                                            period=1 / CONTROL_HZ)
            self.run_segments(tracker, self.plan.pick, recorded=True)
            self.recorded.set()
            hold, zero = self.reference, np.zeros(7)
            while not self.resume.is_set():             # keep holding the block while files are written
                if self.abort.is_set():
                    raise RuntimeError("aborted")
                self.track(tracker, hold, zero)
            self.run_segments(tracker, self.plan.put_back if self.grasped else self.plan.give_up,
                              recorded=False)
        except BaseException as exc:
            self.error = exc
        finally:
            if tracker is not None:
                try:
                    stop_tracker(tracker)
                except Exception as exc:
                    if self.error is None:
                        self.error = exc
            self.recorded.set()
            self.done.set()

    def stop(self):
        self.abort.set()
        self.resume.set()
        if self.ident is not None:
            self.join(timeout=3)
            if self.is_alive():
                self.robot.stop()
                self.join(timeout=3)


def stop_tracker(tracker):
    """End the tracker with a short damping-only stop (franky's default sinks the arm for 2 s)."""
    try:
        from franky import TorqueStopMotion
    except ImportError:
        return tracker.stop()
    return tracker.stop(TorqueStopMotion(ramp_duration=STOP_RAMP_S, max_duration=STOP_MAX_S))


def go_home(robot, JointMotion):
    """Not recorded: franky's position-controlled move to HOME, so every episode starts exactly there."""
    robot.recover_from_errors()
    robot.move(JointMotion(HOME))


def write_episode(handle, records, runner, args, recorders, holding, failure):
    """Shared h5 layout, then replace the waypoint labels with the tracker reference + gripper command."""
    plan = runner.plan
    base.save_h5(handle, records, dict(
        host=args.host, rate_hz=base.RATE_HZ, control_hz=CONTROL_HZ, task="pick up the block", home_q=HOME,
        grasp_pose=plan.grasp, grasp_q=plan.grasp_q, above_pose=plan.above, lift_pose=plan.lifted,
        cameras=[rec.cam.name for rec in recorders], gripper_closed_on_block=runner.grasped,
        holding_after_lift=holding, max_tracking_error_rad=runner.max_seen_error,
        error="" if failure is None else repr(failure), created_utc=datetime.now(timezone.utc).isoformat()))
    traj = handle["traj_0"]
    target = traj["real/target_q"][:]
    phases = traj["obs/extra/policy_phase"][:]
    command = np.array([GRIPPER_COMMAND[PHASES[p]] for p in phases], dtype=np.float32)
    for t in range(len(target)):
        traj["actions/joint_pos"][t] = base.json_row({"arm": target[t].tolist(), "gripper": [float(command[t])]})
        traj["actions/commanded_action"][t] = base.json_row(
            {"arm": target[t].tolist(), "base": [], "gripper": [float(command[t])]})
    traj["real"].create_dataset("gripper_command", data=command)
    traj["actions"].attrs["meaning"] = ("arm: joint reference sent to the impedance tracker at the tick (rad); "
                                        "gripper: commanded 0 open / 1 close")
    traj["obs/extra/policy_phase"].attrs["phases"] = json.dumps(PHASES)
    traj["success"][-1] = failure is None and holding
    traj["fail"][-1] = not traj["success"][-1]
    t0 = records[0]["t0"]
    for rec in recorders:
        rec.write_index(t0, [record[rec.cam.name][0] for record in records])


def collect_episode(robot, gripper_reader, listener, cameras, plan, args, row, control_cpus):
    """Record the pick; return the runner, still holding the block, and whether it holds it."""
    runner = PickRunner(robot, gripper_reader.gripper, plan, args, control_cpus)
    records, recorders = [], []
    failure, holding = None, False
    row.mkdir(parents=True, exist_ok=False)
    with h5py.File(row / "trajectory.h5", "w") as handle:
        try:
            for cam in cameras:
                group = handle.create_group(f"traj_0/real/cameras/{cam.name}")
                rec = base.CameraRecorder(cam, group, row, not args.no_depth, args.rgb_h5)
                rec.begin()
                recorders.append(rec)
            deadline = time.monotonic() + 10
            while any(not rec.number for rec in recorders):
                for rec in recorders:
                    if rec.error is not None:
                        raise RuntimeError(f"{rec.cam.name}: recorder failed: {rec.error}")
                    rec.cam.latest()
                if time.monotonic() > deadline:
                    raise RuntimeError("camera recorders did not receive initial frames")
                time.sleep(0.01)
            t0 = time.monotonic()
            tick = 0
            while True:
                with runner.lock:
                    reference = runner.reference
                    record = base.sample(robot, listener, gripper_reader,
                                         SimpleNamespace(phase=runner.phase, positions={runner.phase: reference}),
                                         recorders, t0)
                record["t0"] = t0
                for rec in recorders:
                    if record[rec.cam.name][1] > 1000:
                        raise RuntimeError(f"{rec.cam.name}: camera frame older than 1 s")
                    if rec.error is not None:
                        raise RuntimeError(f"{rec.cam.name}: recorder failed: {rec.error}")
                records.append(record)
                if runner.ident is None and time.monotonic() - t0 >= args.lead:
                    if all(rec.ready() for rec in recorders):
                        runner.start()
                    elif time.monotonic() - t0 > args.lead + 10:
                        raise RuntimeError("video encoders did not start")
                if runner.recorded.is_set():
                    if runner.error is not None:
                        raise RuntimeError(f"pick motion failed: {runner.error}") from runner.error
                    break
                tick = max(tick + 1, int((time.monotonic() - t0) * base.RATE_HZ) + 1)
                time.sleep(max(0, t0 + tick / base.RATE_HZ - time.monotonic()))
            width = gripper_reader.width
            holding = runner.grasped and np.isfinite(width) and width > args.min_width
        except BaseException as exc:
            failure = exc
            runner.stop()
            raise
        finally:
            try:
                for rec in recorders:
                    rec.finish()
                    if rec.error is not None and failure is None:
                        failure = RuntimeError(f"{rec.cam.name}: recorder failed: {rec.error}")
                if records:
                    write_episode(handle, records, runner, args, recorders, holding, failure)
            finally:
                (row / "result.json").write_text(json.dumps(dict(
                    clean_success=failure is None and holding and bool(records),
                    holding_after_lift=bool(holding), gripper_closed_on_block=runner.grasped,
                    gripper_width_m=float(gripper_reader.width), max_tracking_error_rad=runner.max_seen_error,
                    error="" if failure is None else repr(failure)), indent=2) + "\n")
    print(f"saved {len(records)} ticks: {row}  ({'holding the block' if holding else 'NO BLOCK'}, "
          f"width {gripper_reader.width * 1000:.1f} mm)")
    return runner, holding


def put_back(runner):
    """Not recorded: the runner lowers the block, opens, rises and goes HOME."""
    print("putting the block back (not recorded)")
    runner.resume.set()
    runner.join()
    if runner.error is not None:
        raise RuntimeError(f"put-back failed: {runner.error}") from runner.error


def set_pose(args):
    from franky import Robot

    state = base.read_robot_state(Robot(args.host))
    POSE_FILE.write_text(json.dumps(dict(
        grasp_pose=state["O_T_EE"].tolist(), grasp_q=state["q"].tolist(),
        saved_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    print(f"grasp pose at {np.round(state['translation'], 3).tolist()} m -> {POSE_FILE}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-pose", action="store_true", help="save the arm's current pose as the grasp pose and exit")
    p.add_argument("--plan-only", action="store_true", help="build and print the trajectory, no hardware")
    p.add_argument("--host", default="172.16.0.2")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--approach", type=float, default=0.10, help="m above the grasp pose to come down from")
    p.add_argument("--lift", type=float, default=0.10, help="m to lift the block")
    p.add_argument("--place-gap", type=float, default=0.005,
                   help="m above the grasp height to release the block on put-back")
    p.add_argument("--hold", type=float, default=1.0, help="s to hold the block up, recorded")
    p.add_argument("--jitter", type=float, default=0.0, help="random x/y offset (m) of the approach point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-velocity", type=float, default=0.4, help="peak joint speed of joint moves (rad/s)")
    p.add_argument("--speed", type=float, default=0.05, help="peak hand speed of the straight lines (m/s)")
    p.add_argument("--max-error", type=float, default=0.15, help="abort if a joint lags its reference more (rad)")
    p.add_argument("--dynamics", type=float, default=0.1, help="relative dynamics of the first move HOME")
    p.add_argument("--grasp-force", type=float, default=20.0, help="N")
    p.add_argument("--gripper-speed", type=float, default=0.05, help="m/s")
    p.add_argument("--min-width", type=float, default=0.003, help="m; narrower after the lift = block dropped")
    p.add_argument("--lead", type=float, default=1.0, help="s recorded before the arm moves")
    p.add_argument("--control-cpu", type=int, default=None)
    p.add_argument("--warmup-timeout", type=float, default=5.0)
    p.add_argument("--allow-missing", action="store_true", help="record even if some sensors are silent")
    p.add_argument("--wrist-serial")
    p.add_argument("--exo-serial")
    p.add_argument("--camera", nargs=2, action="append", default=[], metavar=("NAME", "SERIAL"))
    p.add_argument("--no-cameras", action="store_true")
    p.add_argument("--no-camera-reset", action="store_true")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--rgb-h5", action="store_true")
    p.add_argument("--output", type=Path, help="new batch directory (default recordings/pick_<utc stamp>)")
    p.add_argument("--sim", action="store_true",
                   help="run on a MuJoCo arm + block instead of the robot (desktop, needs the parent repo)")
    p.add_argument("--sim-block", type=float, default=0.05, help="simulated cube size (m); 0 = no block")
    p.add_argument("--sim-view", action="store_true", help="open the MuJoCo viewer")
    live_dashboard.add_args(p)
    args = p.parse_args(argv)
    if args.sim:
        args.no_cameras, args.allow_missing, args.warmup_timeout = True, True, min(args.warmup_timeout, 0.5)
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if not 0 < args.dynamics <= 1:
        p.error("--dynamics must be in (0, 1]")
    if args.approach < 0.02 or args.lift < 0.02 or args.jitter < 0:
        p.error("--approach and --lift must be at least 0.02 m, --jitter nonnegative")
    if not 0 <= args.place_gap < min(args.approach, args.lift):
        p.error("--place-gap must be nonnegative and below --approach and --lift")
    if args.max_velocity <= 0 or args.speed <= 0 or args.hold < 0:
        p.error("--max-velocity and --speed must be positive, --hold nonnegative")
    if not (args.set_pose or args.plan_only or args.no_cameras):
        try:
            base.camera_specs(args)
        except ValueError as exc:
            p.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.set_pose:
        return set_pose(args)
    if not POSE_FILE.exists():
        raise SystemExit(f"no grasp pose yet: {POSE_FILE} missing; run python data.py pick --set-pose first")
    saved = json.loads(POSE_FILE.read_text())
    rng = np.random.default_rng(args.seed)
    plans = [Plan(saved, args, rng) for _ in range(args.episodes)]     # all checked before the arm moves
    print(f"episode 1 trajectory:\n{plans[0].summary()}")
    if args.plan_only:
        return
    world = None
    if args.sim:
        from . import sim_franky
        world = sim_franky.install(saved, HOME, args.sim_block, args.sim_view)
    stamp = datetime.now(timezone.utc).strftime("pick_sim_%Y%m%dT%H%M%SZ" if args.sim else "pick_%Y%m%dT%H%M%SZ")
    output = args.output or Path("recordings") / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(dict(vars(args), **saved), default=str, indent=2) + "\n")

    from franky import JointMotion, Robot

    affinity = os.sched_getaffinity(0)
    with ExitStack() as cleanup:
        if world is not None:
            cleanup.callback(world.join, 2.0)
            cleanup.callback(world.stop_event.set)
        cleanup.callback(os.sched_setaffinity, 0, affinity)
        control_cpus = base.reserve_control_cpus(args.control_cpu)
        sockets = base.open_sensor_sockets()
        for sock in sockets:
            cleanup.callback(sock.close)
        listener = base.SensorListener(sockets)
        listener.start()
        cleanup.callback(listener.join, 1.0)
        cleanup.callback(listener.stop_event.set)
        base.wait_for_sensors(listener, args.warmup_timeout, args.allow_missing)
        cameras = [] if args.no_cameras else base.open_cameras(args)
        for cam in cameras:
            cleanup.callback(cam.close)
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = args.dynamics
        gripper = base.GripperReader(args.host)
        gripper.start()
        cleanup.callback(gripper.stop_event.set)
        cleanup.callback(live_dashboard.stop, live_dashboard.start(
            args, robot, gripper, listener, cameras, PHASES, "pick" + (" (sim)" if args.sim else "")))
        print("going home, gripper open (not recorded)")
        gripper.gripper.open(args.gripper_speed)
        for i, plan in enumerate(plans):
            print(f"episode {i + 1}/{args.episodes}")
            go_home(robot, JointMotion)
            if world is not None:
                start = world.block_position()
                world.reset_block_stats()
            runner, holding = collect_episode(robot, gripper, listener, cameras, plan, args,
                                              output / f"episode_{i:04d}", control_cpus)
            put_back(runner)
            if world is not None and start is not None:
                end = world.block_position()
                first = start if i == 0 else first
                print(f"sim: block rose {(world.block_top - start[2]) * 1000:.0f} mm, put back "
                      f"{np.linalg.norm(end - start) * 1000:.1f} mm from its start ({np.round((end - first) * 1000, 1).tolist()} mm "
                      f"from episode 1); "
                      f"max tracking error {runner.max_seen_error:.4f} rad, "
                      f"peak torque {np.round(world.max_torque, 1).tolist()} Nm")
            if not holding:
                go_home(robot, JointMotion)
                raise SystemExit("pick failed; check the block position, then run again "
                                 "(re-run --set-pose if the spot moved)")
        go_home(robot, JointMotion)                 # end held at HOME, not where the last stop left it
    print(f"done: {args.episodes} episodes in {output}")
