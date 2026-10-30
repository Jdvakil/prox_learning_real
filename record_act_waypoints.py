"""Collect simple A -> B ACT episodes with smooth, noisy joint position references.

Each row has the same RGB-D/proximity/HDF5 layout as record_dataset.py. Reset to A
is outside the episode. Actions are absolute joint references in radians, sampled
from the reference actually sent to a 100 Hz JointImpedanceTracker, not final goals.
The gripper is fixed open. Default: two episodes; use --episodes 20 for a batch.
--dry-run writes reproducible reference plans without connecting to hardware.
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

import record_dataset as base
from kinesthetic import Q_MIN, Q_MAX, REPLAY_STIFFNESS, REPLAY_DAMPING


START = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]
GOAL = [0.2, -0.1, 0.1, -2.1, 0.0, 2.1, 0.7]
CONTROL_HZ = 100.0


class ReferencePlan:
    """Minimum-jerk interpolation plus a differentiable Gaussian perturbation.

Three sine modes per joint have independent Gaussian coefficients. A smooth
envelope makes position, velocity and acceleration perturbations zero at A/B.
noise_std is coefficient scale in radians, not independent noise at every tick.
"""

    def __init__(self, start, goal, duration=8.0, noise_std=0.01, seed=0):
        self.start, self.goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        self.duration, self.noise_std, self.seed = duration, noise_std, seed
        if self.start.shape != (7,) or self.goal.shape != (7,):
            raise ValueError("start and goal must each contain seven joints")
        if not np.isfinite([duration, noise_std]).all() or duration <= 0 or noise_std < 0:
            raise ValueError("duration must be positive and noise-std nonnegative and finite")
        if not np.isfinite([self.start, self.goal]).all():
            raise ValueError("joint positions must be finite")
        self.coefficients = np.random.default_rng(seed).normal(0, noise_std / np.sqrt(3), (3, 7))

    def evaluate(self, t):
        u = np.clip(np.asarray(t, dtype=float) / self.duration, 0, 1)
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        dblend = 30 * u**2 * (1 - u)**2
        envelope = 64 * u**3 * (1 - u)**3
        denvelope = 192 * u**2 * (1 - u)**2 * (1 - 2 * u)
        omega = np.pi * np.arange(1, 4)
        angles = u[..., None] * omega
        wave = np.sin(angles) @ self.coefficients
        dwave = (np.cos(angles) * omega) @ self.coefficients
        nominal = self.start + blend[..., None] * (self.goal - self.start)
        noise = envelope[..., None] * wave
        q = nominal + noise
        dq = (dblend[..., None] * (self.goal - self.start)
              + denvelope[..., None] * wave + envelope[..., None] * dwave) / self.duration
        return q, dq, nominal, noise

    def validate(self, max_velocity=0.25, max_acceleration=0.5, max_noise=0.05):
        # Dense preflight, also saved by --dry-run for inspection. Not a collision check.
        t = np.linspace(0, self.duration, max(1001, int(self.duration * 1000) + 1))
        q, dq, _, noise = self.evaluate(t)
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError("reference contains non-finite positions or velocities")
        ddq = np.gradient(dq, t, axis=0)
        if np.any(q < np.asarray(Q_MIN) + 0.05) or np.any(q > np.asarray(Q_MAX) - 0.05):
            raise ValueError("reference violates FR3 joint limits with 0.05 rad margin")
        for value, limit, name in ((dq, max_velocity, "velocity (rad/s)"),
                                   (ddq, max_acceleration, "acceleration (rad/s^2)"),
                                   (noise, max_noise, "noise (rad)")):
            peak = float(np.abs(value).max())
            if peak > limit:
                raise ValueError(f"reference {name} peak {peak:.4f} exceeds {limit}; "
                                 "increase --duration or reduce --noise-std / motion range")

    def save(self, path):
        t = np.linspace(0, self.duration, int(np.ceil(self.duration * CONTROL_HZ)) + 1)
        q, dq, nominal, noise = self.evaluate(t)
        np.savez(path, t=t, q_ref=q, dq_ref=dq, nominal_q=nominal, noise_q=noise,
                 coefficients=self.coefficients, seed=self.seed, start=self.start, goal=self.goal)


class ReferenceRunner(threading.Thread):
    def __init__(self, robot, plan, args, control_cpus):
        super().__init__(daemon=True)
        self.robot, self.plan, self.args, self.control_cpus = robot, plan, args, control_cpus
        self.lock = threading.Lock()
        self.abort, self.done = threading.Event(), threading.Event()
        self.error = None
        self.reference = (plan.start.copy(), np.zeros(7), plan.start.copy(), np.zeros(7), 0.0)
        self.commands = []
        self.max_seen_error = 0.0

    def run(self):
        try:
            from franky import JointImpedanceTracker

            if self.control_cpus:
                os.sched_setaffinity(0, self.control_cpus)
            with JointImpedanceTracker(self.robot, stiffness=REPLAY_STIFFNESS,
                                       damping=REPLAY_DAMPING, lower_joint_limits=Q_MIN,
                                       upper_joint_limits=Q_MAX, period=1 / CONTROL_HZ) as tracker:
                start = time.monotonic()
                while not self.abort.is_set():
                    if tracker.tick() is None:
                        raise RuntimeError("robot left impedance control")
                    now = time.monotonic()
                    clock = min(now - start, self.plan.duration)
                    q, dq, nominal, noise = self.plan.evaluate(clock)
                    with self.lock:
                        tracker.set_target(q, dq)
                        sent_at = time.monotonic()
                        self.reference = (q, dq, nominal, noise, clock)
                        self.commands.append((sent_at, q, dq))
                    error = float(np.abs(np.asarray(self.robot.state.q) - q).max())
                    self.max_seen_error = max(self.max_seen_error, error)
                    if error > self.args.max_error:
                        raise RuntimeError(f"tracking error {error:.4f} rad > {self.args.max_error}")
                    if now - start >= self.plan.duration + self.args.tail:
                        if error > self.args.goal_tolerance:
                            raise RuntimeError(f"goal error {error:.4f} rad > {self.args.goal_tolerance}")
                        return
                raise RuntimeError("episode aborted")
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()

    def stop(self):
        self.abort.set()
        if self.ident is not None:
            self.join(timeout=3)
            if self.is_alive():
                self.robot.stop()
                self.join(timeout=3)
            if self.is_alive():
                raise RuntimeError("reference thread did not stop")


def sample(robot, listener, gripper, runner, recorders, t0):
    # Freeze reference updates while reading observations, so the label is the
    # active command, not a separately recomputed future or final-goal reference.
    with runner.lock:
        q, dq, nominal, noise, clock = runner.reference
        record = base.sample(robot, listener, gripper,
                             SimpleNamespace(phase=0, positions=[q]), recorders, t0)
        record.update(dq_ref=dq.copy(), nominal_q=nominal.copy(), noise_q=noise.copy(), clock=clock)
    for rec in recorders:
        if record[rec.cam.name][1] > 1000:
            raise RuntimeError(f"{rec.cam.name}: camera frame older than 1 s")
        if rec.error is not None:
            raise RuntimeError(f"{rec.cam.name}: recorder failed: {rec.error}")
    return record


def write_episode(handle, records, runner, args, plan, recorders, t0, failure):
    metadata = dict(host=args.host, rate_hz=base.RATE_HZ, policy_dt_ms=1000 / base.RATE_HZ,
                    control_hz=CONTROL_HZ, start_q=plan.start, goal_q=plan.goal,
                    cameras=[rec.cam.name for rec in recorders],
                    duration_s=plan.duration, noise_std_rad=plan.noise_std, seed=plan.seed,
                    error="" if failure is None else repr(failure),
                    created_utc=datetime.now(timezone.utc).isoformat())
    base.save_h5(handle, records, metadata)
    traj = handle["traj_0"]
    traj["actions"].attrs["meaning"] = (
        "absolute joint reference actually sent to impedance tracker at observation time; "
        "radians, fixed open gripper (0); no action time shift applied")
    for key in ("dq_ref", "nominal_q", "noise_q", "clock"):
        traj["real"].create_dataset(key, data=np.stack([record[key] for record in records]))
    traj["real"].create_dataset("noise_coefficients", data=plan.coefficients)
    commands = traj["real"].create_group("control")
    commands.create_dataset("t", data=[c[0] - t0 for c in runner.commands])
    commands.create_dataset("q_ref", data=np.asarray([c[1] for c in runner.commands]).reshape(-1, 7))
    commands.create_dataset("dq_ref", data=np.asarray([c[2] for c in runner.commands]).reshape(-1, 7))
    traj["success"][-1] = failure is None
    traj["fail"][-1] = failure is not None
    for rec in recorders:
        rec.write_index(t0, [record[rec.cam.name][0] for record in records])


def collect_episode(robot, listener, gripper, cameras, plan, args, row, control_cpus):
    from franky import JointMotion

    print(f"reset to A (not recorded): {plan.start.tolist()}")
    robot.move(JointMotion(plan.start.tolist()))
    time.sleep(args.reset_settle)
    if np.max(np.abs(base.read_robot_state(robot)["q"] - plan.start)) > args.goal_tolerance:
        raise RuntimeError("reset did not settle at A")
    row.mkdir(parents=True, exist_ok=False)
    plan.save(row / "reference_plan.npz")
    runner = ReferenceRunner(robot, plan, args, control_cpus)
    records, recorders = [], []
    failure = None
    t0 = time.monotonic()
    with h5py.File(row / "trajectory.h5", "w") as handle:
        try:
            for cam in cameras:
                group = handle.create_group(f"traj_0/real/cameras/{cam.name}")
                rec = base.CameraRecorder(cam, group, row, not args.no_depth, args.rgb_h5)
                rec.begin()
                recorders.append(rec)
            t0 = time.monotonic()
            tick = 0
            while True:
                records.append(sample(robot, listener, gripper, runner, recorders, t0))
                if runner.ident is None and time.monotonic() - t0 >= args.lead:
                    if all(rec.ready() for rec in recorders):
                        runner.start()
                    elif time.monotonic() - t0 > args.lead + 10:
                        raise RuntimeError("video encoders did not start")
                if runner.done.is_set():
                    if runner.error is not None:
                        raise RuntimeError(f"reference motion failed: {runner.error}") from runner.error
                    break
                tick = max(tick + 1, int((time.monotonic() - t0) * base.RATE_HZ) + 1)
                time.sleep(max(0, t0 + tick / base.RATE_HZ - time.monotonic()))
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                try:
                    runner.stop()
                except BaseException as exc:
                    failure = failure or exc
                for rec in recorders:
                    rec.finish()
                    if rec.error is not None and failure is None:
                        failure = RuntimeError(f"{rec.cam.name}: recorder failed: {rec.error}")
                if records:
                    write_episode(handle, records, runner, args, plan, recorders, t0, failure)
            except BaseException as exc:
                failure = failure or exc
                raise
            finally:
                (row / "result.json").write_text(json.dumps(dict(
                    clean_success=failure is None and bool(records),
                    error="" if failure is None else repr(failure), seed=plan.seed,
                    start_q=plan.start.tolist(), goal_q=plan.goal.tolist(),
                    max_tracking_error_rad=runner.max_seen_error), indent=2) + "\n")
    if failure is not None:
        raise failure
    print(f"saved {len(records)} ticks: {row}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.16.0.2")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--start", type=float, nargs=7, default=START, metavar="Q")
    parser.add_argument("--goal", type=float, nargs=7, default=GOAL, metavar="Q")
    parser.add_argument("--duration", type=float, default=8.0, help="A -> B seconds")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian mode scale in radians")
    parser.add_argument("--seed", type=int, default=0, help="episode seeds are seed + episode index")
    parser.add_argument("--max-velocity", type=float, default=0.25, help="reference preflight rad/s")
    parser.add_argument("--max-acceleration", type=float, default=0.5, help="reference preflight rad/s^2")
    parser.add_argument("--max-noise", type=float, default=0.05, help="reject larger perturbations (rad)")
    parser.add_argument("--max-error", type=float, default=0.15, help="abort tracking error (rad)")
    parser.add_argument("--goal-tolerance", type=float, default=0.02, help="start/goal acceptance (rad)")
    parser.add_argument("--dynamics", type=float, default=0.05, help="reset JointMotion dynamics")
    parser.add_argument("--lead", type=float, default=1.0)
    parser.add_argument("--tail", type=float, default=1.0, help="hold B for this many seconds")
    parser.add_argument("--reset-settle", type=float, default=0.5)
    parser.add_argument("--control-cpu", type=int, default=None)
    parser.add_argument("--warmup-timeout", type=float, default=5.0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--wrist-serial")
    parser.add_argument("--exo-serial")
    parser.add_argument("--camera", nargs=2, action="append", default=[], metavar=("NAME", "SERIAL"))
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-camera-reset", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--rgb-h5", action="store_true")
    parser.add_argument("--no-gripper", action="store_true", help="skip opening/reading; keep gripper open manually")
    parser.add_argument("--dry-run", action="store_true", help="save plans only; no hardware connection")
    parser.add_argument("--output", type=Path, default=None, help="new batch directory containing episode rows")
    args = parser.parse_args(argv)
    try:
        if args.episodes <= 0 or args.seed < 0:
            raise ValueError("episodes must be positive and seed nonnegative")
        for key in ("duration", "max_velocity", "max_acceleration", "max_noise", "max_error",
                    "goal_tolerance", "tail", "warmup_timeout"):
            if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
                raise ValueError(f"--{key.replace('_', '-')} must be positive and finite")
        for key in ("lead", "reset_settle", "noise_std"):
            if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
                raise ValueError(f"--{key.replace('_', '-')} must be nonnegative and finite")
        if not 0 < args.dynamics <= 1:
            raise ValueError("--dynamics must be in (0, 1]")
        if not args.no_cameras and not args.dry_run:
            base.camera_specs(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    plans = [ReferencePlan(args.start, args.goal, args.duration, args.noise_std, args.seed + i)
             for i in range(args.episodes)]
    for plan in plans:
        plan.validate(args.max_velocity, args.max_acceleration, args.max_noise)
    output = args.output or Path("recordings") / datetime.now(timezone.utc).strftime("act_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    if args.dry_run:
        for i, plan in enumerate(plans):
            plan.save(output / f"episode_{i:04d}_plan.npz")
        print(f"validated {len(plans)} plans; no hardware connected: {output}")
        return

    from franky import Gripper, Robot

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
        gripper = None
        if not args.no_gripper:
            if not Gripper(args.host).open(0.05):
                raise RuntimeError("gripper did not open")
            gripper = base.GripperReader(args.host)
            gripper.start()
            cleanup.callback(gripper.stop_event.set)
        for i, plan in enumerate(plans):
            print(f"episode {i + 1}/{len(plans)}, seed {plan.seed}")
            collect_episode(robot, listener, gripper, cameras, plan, args,
                            output / f"episode_{i:04d}", control_cpus)


if __name__ == "__main__":
    main()
