"""Scripted block pick: go over a fixed block spot, close the gripper, lift, put the block back.

No model and no vision. The block has to sit where you showed the robot, every episode.

  1. Put the block at its spot. Move the arm by hand (Desk guiding mode) so the open fingers
     are around the block, then, with FCI active and the arm still:
         python data.py pick --set-pose                  # saves pick_pose.json (no motion)
  2. Collect:
         python data.py pick --episodes 10 --wrist-serial <serial> --exo-serial <serial>
                                                         # -> recordings/pick_<utc stamp>/episode_0000/ ...

Each episode is recorded: the arm starts at HOME with the gripper open, moves --approach m above
the grasp pose, goes down, closes the gripper, lifts --lift m and holds --hold s. Then, not
recorded, it lowers the block back onto its spot, opens, backs off and returns HOME, so the next
episode finds the block where it was. --jitter adds a random x/y offset (m) to the point above
the block that the arm approaches from; the grasp pose itself never changes.

Rows use the record_dataset.py layout (trajectory.h5 + videos + result.json). Actions are the
measured arm position at the next 15 Hz tick plus the gripper command (0 open, 1 close); the
waypoint index is obs/extra/policy_phase (names in its "phases" attribute). An episode whose
gripper does not hold the block after the lift is saved with clean_success false and stops the
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

import h5py
import numpy as np

from . import record_dataset as base

HOME = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]
POSE_FILE = Path(__file__).resolve().parents[1] / "pick_pose.json"
PHASES = ["start", "approach", "descend", "grasp", "lift", "hold"]
GRIPPER_COMMAND = {"start": 0.0, "approach": 0.0, "descend": 0.0, "grasp": 1.0, "lift": 1.0, "hold": 1.0}


def shifted(matrix, dx=0.0, dy=0.0, dz=0.0):
    out = np.array(matrix, dtype=np.float64)
    out[:3, 3] += [dx, dy, dz]
    return out


class PickRunner(threading.Thread):
    """Runs one pick in the background while the main loop logs at 15 Hz."""

    def __init__(self, robot, gripper, grasp_pose, args, rng, control_cpus=None):
        super().__init__(daemon=True)
        self.robot, self.gripper, self.args, self.control_cpus = robot, gripper, args, control_cpus
        jitter = rng.uniform(-args.jitter, args.jitter, 2) if args.jitter > 0 else np.zeros(2)
        self.grasp = np.asarray(grasp_pose, dtype=np.float64)
        self.above = shifted(self.grasp, jitter[0], jitter[1], args.approach)
        self.lifted = shifted(self.grasp, dz=args.lift)
        self.phase = 0
        self.positions = np.full((len(PHASES), 7), np.nan)     # base.sample stores this as target_q
        self.grasped = False
        self.error = None
        self.abort, self.done = threading.Event(), threading.Event()

    def enter(self, name):
        self.phase = PHASES.index(name)
        print(f"  {name}")

    def run(self):
        from franky import Affine, CartesianMotion

        if self.control_cpus:
            os.sched_setaffinity(0, self.control_cpus)
        try:
            for name, target in (("approach", self.above), ("descend", self.grasp)):
                if self.abort.is_set():
                    return
                self.enter(name)
                self.robot.move(CartesianMotion(Affine(target)))
            self.enter("grasp")
            # Unknown block width: any closed width counts (epsilon_outer = 1 m).
            self.grasped = bool(self.gripper.grasp(0.0, self.args.gripper_speed, self.args.grasp_force,
                                                   epsilon_inner=0.005, epsilon_outer=1.0))
            if self.abort.is_set():
                return
            self.enter("lift")
            self.robot.move(CartesianMotion(Affine(self.lifted)))
            self.enter("hold")
            self.abort.wait(self.args.hold)
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()


def write_episode(handle, records, runner, args, recorders, holding, failure):
    """Shared h5 layout, then replace the waypoint labels with next-tick q + gripper command."""
    base.save_h5(handle, records, dict(
        host=args.host, rate_hz=base.RATE_HZ, task="pick up the block", home_q=HOME,
        grasp_pose=runner.grasp, above_pose=runner.above, lift_pose=runner.lifted,
        cameras=[rec.cam.name for rec in recorders], gripper_closed_on_block=runner.grasped,
        holding_after_lift=holding, error="" if failure is None else repr(failure),
        created_utc=datetime.now(timezone.utc).isoformat()))
    traj = handle["traj_0"]
    q = traj["real/q"][:]
    next_q = np.concatenate([q[1:], q[-1:]])
    phases = traj["obs/extra/policy_phase"][:]
    command = np.array([GRIPPER_COMMAND[PHASES[p]] for p in phases], dtype=np.float32)
    for t in range(len(q)):
        traj["actions/joint_pos"][t] = base.json_row({"arm": next_q[t].tolist(), "gripper": [float(command[t])]})
        traj["actions/commanded_action"][t] = base.json_row(
            {"arm": next_q[t].tolist(), "base": [], "gripper": [float(command[t])]})
    traj["real/target_q"][...] = next_q
    traj["real"].create_dataset("gripper_command", data=command)
    traj["actions"].attrs["meaning"] = ("arm: measured joint position at the next 15 Hz tick (rad); "
                                        "gripper: commanded 0 open / 1 close")
    traj["obs/extra/policy_phase"].attrs["phases"] = json.dumps(PHASES)
    traj["success"][-1] = failure is None and holding
    traj["fail"][-1] = not traj["success"][-1]
    t0 = records[0]["t0"]
    for rec in recorders:
        rec.write_index(t0, [record[rec.cam.name][0] for record in records])


def put_back(robot, gripper, runner, args):
    """Not recorded: lower the block onto its spot, open, back off, go home."""
    from franky import Affine, CartesianMotion, JointMotion

    robot.recover_from_errors()
    if runner.grasped:
        robot.move(CartesianMotion(Affine(runner.grasp)))
    gripper.open(args.gripper_speed)
    robot.move(CartesianMotion(Affine(shifted(runner.grasp, dz=args.approach))))
    robot.move(JointMotion(HOME))


def collect_episode(robot, gripper_reader, listener, cameras, grasp_pose, args, row, rng, control_cpus):
    runner = PickRunner(robot, gripper_reader.gripper, grasp_pose, args, rng, control_cpus)
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
                record = base.sample(robot, listener, gripper_reader, runner, recorders, t0)
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
                if runner.done.is_set():
                    if runner.error is not None:
                        raise RuntimeError(f"pick motion failed: {runner.error}") from runner.error
                    break
                tick = max(tick + 1, int((time.monotonic() - t0) * base.RATE_HZ) + 1)
                time.sleep(max(0, t0 + tick / base.RATE_HZ - time.monotonic()))
            width = gripper_reader.width
            holding = runner.grasped and np.isfinite(width) and width > args.min_width
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                for rec in recorders:
                    rec.finish()
                    if rec.error is not None and failure is None:
                        failure = RuntimeError(f"{rec.cam.name}: recorder failed: {rec.error}")
                if runner.is_alive():
                    base.stop_motion(robot, runner)
                if records:
                    write_episode(handle, records, runner, args, recorders, holding, failure)
            finally:
                (row / "result.json").write_text(json.dumps(dict(
                    clean_success=failure is None and holding and bool(records),
                    holding_after_lift=bool(holding), gripper_closed_on_block=runner.grasped,
                    gripper_width_m=float(gripper_reader.width),
                    error="" if failure is None else repr(failure)), indent=2) + "\n")
    print(f"saved {len(records)} ticks: {row}  ({'holding the block' if holding else 'NO BLOCK'}, "
          f"width {gripper_reader.width * 1000:.1f} mm)")
    return runner, holding


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
    p.add_argument("--host", default="172.16.0.2")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--approach", type=float, default=0.10, help="m above the grasp pose to come down from")
    p.add_argument("--lift", type=float, default=0.10, help="m to lift the block")
    p.add_argument("--hold", type=float, default=1.0, help="s to hold the block up, recorded")
    p.add_argument("--jitter", type=float, default=0.0, help="random x/y offset (m) of the approach point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dynamics", type=float, default=0.1, help="relative velocity / acceleration / jerk")
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
    args = p.parse_args(argv)
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if not 0 < args.dynamics <= 1:
        p.error("--dynamics must be in (0, 1]")
    if args.approach < 0.02 or args.lift < 0.02 or args.jitter < 0:
        p.error("--approach and --lift must be at least 0.02 m, --jitter nonnegative")
    if not args.set_pose and not args.no_cameras:
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
    grasp_pose = np.asarray(json.loads(POSE_FILE.read_text())["grasp_pose"])
    output = args.output or Path("recordings") / datetime.now(timezone.utc).strftime("pick_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(dict(vars(args), grasp_pose=grasp_pose.tolist()),
                                                   default=str, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)

    from franky import JointMotion, Robot

    affinity = os.sched_getaffinity(0)
    with ExitStack() as cleanup:
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
        print("going home, gripper open (not recorded)")
        gripper.gripper.open(args.gripper_speed)
        robot.move(JointMotion(HOME))
        for i in range(args.episodes):
            print(f"episode {i + 1}/{args.episodes}")
            runner, holding = collect_episode(robot, gripper, listener, cameras, grasp_pose, args,
                                              output / f"episode_{i:04d}", rng, control_cpus)
            print("putting the block back (not recorded)")
            put_back(robot, gripper.gripper, runner, args)
            if not holding:
                raise SystemExit("pick failed; check the block position, then run again "
                                 "(re-run --set-pose if the spot moved)")
    print(f"done: {args.episodes} episodes in {output}")
