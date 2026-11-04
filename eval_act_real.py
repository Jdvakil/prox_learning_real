"""Evaluate vanilla RGB + joint-state ACT on the real FR3 using Franky.

No gripper is connected or commanded. Supports the local ACT state_dict checkpoints,
dataset_stats.pkl and saved training metadata. --offline-row runs inference on a
recorded episode without importing Franky or connecting to hardware.
"""

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import threading
import time

import h5py
import numpy as np

import record_dataset as base
from record_act_waypoints import START, GOAL, CONTROL_HZ
from kinesthetic import Q_MIN, Q_MAX, REPLAY_STIFFNESS, REPLAY_DAMPING

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "models/act_test_20"


def read_training_config(directory, explicit=None):
    if explicit:
        data = json.loads(explicit.read_text())
        return data.get("policy_config", data)
    path = directory / "training_config.json"
    if path.exists():
        data = json.loads(path.read_text())
        return dict(data.get("arguments", {}), **data["policy_config"])
    candidates = [directory / "wandb/latest-run/files/config.yaml"]
    candidates.extend(sorted(directory.glob("wandb/run-*/files/config.yaml")))
    for path in candidates:
        if path.exists():
            import yaml
            data = yaml.safe_load(path.read_text())
            return {k: v["value"] for k, v in data.items()
                    if k != "_wandb" and isinstance(v, dict) and "value" in v}
    return {}


def state_vector(q, dimension):
    if dimension not in (7, 9):
        raise ValueError(f"unsupported state dimension {dimension}; expected arm7 or arm7 + two zero pads")
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("joint observation must contain seven finite positions")
    result = np.zeros(dimension, dtype=np.float32)
    result[:7] = q
    return result


class ACTInference:
    """Inference-only ACT builder: no optimizer, implicit CUDA, downloads or argv parsing."""

    def __init__(self, args, proximity_encoder=None):
        import torch

        self.torch = torch
        torch.set_num_threads(args.torch_threads)
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.config = read_training_config(args.ckpt_dir, args.policy_config)
        self.proximity_encoder = proximity_encoder
        if proximity_encoder is None and ((args.ckpt_dir / "prox_config.json").exists() or self.config.get("n_proximity_sensors", 0)):
            raise ValueError("this evaluator supports vanilla RGB ACT; PACT needs its trained proximity encoder")
        checkpoint = args.ckpt_dir / args.ckpt_name
        with checkpoint.open("rb") as f:
            if f.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError("checkpoint is a Git LFS pointer; run git lfs pull or copy the actual model file")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "model.query_embed.weight" not in state:
            raise ValueError("expected a local ACTPolicy state_dict checkpoint")
        self.state_dim = int(state["model.input_proj_robot_state.weight"].shape[1])
        self.action_dim = int(state["model.action_head.weight"].shape[0])
        self.chunk_size = int(state["model.query_embed.weight"].shape[0])
        if self.state_dim not in (7, 9) or self.action_dim not in (7, 8):
            raise ValueError(f"unsupported arm-only checkpoint dimensions: {self.state_dim}/{self.action_dim}")
        self.camera_names = list(args.camera_names or self.config.get("camera_names", []))
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("provide unique --camera-names in exact training order, or saved training metadata")
        data_dir = args.data_dir
        meta_path = data_dir / "convert_meta.json" if data_dir else None
        meta = json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else {}
        packaged_meta = args.ckpt_dir / "image_metadata.json"
        if not meta and packaged_meta.exists():
            meta = json.loads(packaged_meta.read_text())
        self.height = args.image_height or meta.get("image_h")
        self.width = args.image_width or meta.get("image_w")
        if not self.height or not self.width:
            raise ValueError("training image size missing: pass --data-dir or --image-height and --image-width")
        if meta.get("camera_names") and self.camera_names != meta["camera_names"]:
            raise ValueError("camera order differs from converted dataset metadata")
        stats_path = args.ckpt_dir / "dataset_stats.npz"
        if stats_path.exists():
            with np.load(stats_path, allow_pickle=False) as f:
                stats = {key: f[key].copy() for key in f.files}
        else:
            with (args.ckpt_dir / "dataset_stats.pkl").open("rb") as f:
                try:
                    stats = pickle.load(f)
                except ModuleNotFoundError as exc:
                    raise ValueError("legacy pickled stats require the training NumPy environment; "
                                     "export a portable model bundle with export_act_model.py") from exc
        self.stats = {k: np.asarray(stats[k], dtype=np.float32) for k in
                      ("qpos_mean", "qpos_std", "action_mean", "action_std")}
        for name, dimension in (("qpos", self.state_dim), ("action", self.action_dim)):
            for suffix in ("mean", "std"):
                value = self.stats[f"{name}_{suffix}"]
                if value.shape != (dimension,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid normalization {name}_{suffix}")
            if np.any(self.stats[f"{name}_std"] <= 0):
                raise ValueError(f"normalization {name}_std must be positive")
            if np.any(abs(self.stats[f"{name}_mean"][7:]) > 1e-6):
                raise ValueError("checkpoint was trained with active extra joints; cannot treat them as zero padding")
        from act_model import build_model
        model_args = argparse.Namespace(**self.config)
        model_args.hidden_dim = int(state["model.query_embed.weight"].shape[1])
        model_args.dim_feedforward = int(state["model.transformer.encoder.layers.0.linear1.weight"].shape[0])
        model_args.enc_layers = len({k.split(".")[3] for k in state if k.startswith("model.encoder.layers.")})
        model_args.dec_layers = len({k.split(".")[4] for k in state if k.startswith("model.transformer.decoder.layers.")})
        model_args.nheads = args.nheads or self.config.get("nheads")
        if not model_args.nheads:
            raise ValueError("attention head count is not encoded in weight shapes: supply --nheads or training metadata")
        model_args.state_dim, model_args.action_dim = self.state_dim, self.action_dim
        model_args.num_queries = self.chunk_size
        model_args.camera_names = self.camera_names
        model_args.n_proximity_sensors = self.config.get("n_proximity_sensors",0) if proximity_encoder is not None else 0
        # The local builder never downloads weights or imports training infrastructure.
        model = build_model(vars(model_args))
        model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
        self.model = model.to(self.device).eval()
        if proximity_encoder is not None:
            self.proximity_encoder.to(self.device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 1, 3, 1, 1)
        self.query_every = 1 if args.temporal_agg else (args.query_every or self.chunk_size)
        if not 1 <= self.query_every <= self.chunk_size:
            raise ValueError("query-every must be between 1 and checkpoint chunk size")
        print(f"ACT: state={self.state_dim}, action={self.action_dim}, chunk={self.chunk_size}, "
              f"query_every={self.query_every}, cameras={self.camera_names}, "
              f"RGB={self.width}x{self.height}, device={self.device}")

    def prepare_images(self, images):
        import cv2
        frames = []
        for name in self.camera_names:
            rgb = np.asarray(images[name])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"{name}: expected uint8 RGB image")
            if rgb.shape[:2] != (self.height, self.width):
                rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        return np.stack(frames)

    def predict(self, q, rgb, proximity=None):
        torch = self.torch
        state = state_vector(q, self.state_dim)
        state = (state - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        images = np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))
        with torch.inference_mode():
            state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            image_t = torch.as_tensor(images, device=self.device).float().unsqueeze(0) / 255
            features = None
            if self.proximity_encoder is not None:
                if proximity is None or np.shape(proximity)!=(40,32,8,8):
                    raise ValueError("PC-ACT requires a (40,32,8,8) causal depth history in meters")
                features = self.proximity_encoder(torch.as_tensor(proximity,device=self.device).float().unsqueeze(0))
            a, _, _ = self.model(state_t, (image_t - self.mean) / self.std, None,proximity_positions=features)
            a = a[0].detach().cpu().numpy()
        if a.shape != (self.chunk_size, self.action_dim) or not np.isfinite(a).all():
            raise ValueError("ACT produced invalid action chunk")
        return a * self.stats["action_std"] + self.stats["action_mean"]


class ChunkPlayer:
    """Chunk execution or overlapping chunk aggregation; zero is a valid prediction."""
    def __init__(self, query_every, temporal=False, decay=0.01):
        self.query_every, self.temporal, self.decay = query_every, temporal, decay
        self.chunks = []

    def needs_query(self, tick):
        return not self.chunks or tick % self.query_every == 0

    def add(self, tick, chunk):
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] not in (7, 8) or not np.isfinite(chunk).all():
            raise ValueError("invalid action chunk")
        self.chunks = [(s, c) for s, c in self.chunks if s + len(c) > tick] if self.temporal else []
        self.chunks.append((tick, chunk.copy()))

    def action(self, tick):
        valid = [(s, c[tick-s]) for s, c in self.chunks if s <= tick < s + len(c)]
        if not valid:
            raise RuntimeError("action chunk exhausted")
        if not self.temporal:
            return valid[-1][1].copy()
        # Explicit recent-prediction preference; do not filter zero-valued actions.
        weights = np.exp(-self.decay * np.asarray([tick-s for s, _ in valid]))
        return np.average(np.stack([a for _, a in valid]), axis=0, weights=weights)

    def next_query_tick(self, prefetch_steps):
        start, chunk = self.chunks[-1]
        ahead = min(prefetch_steps, len(chunk)-1)
        return min(start+self.query_every, start+len(chunk)-ahead)


def predict_timed(policy, q, rgb, proximity=None):
    since = time.monotonic()
    result = policy.predict(q,rgb) if proximity is None else policy.predict(q,rgb,proximity)
    return result, time.monotonic()-since


def checked_target(action, observed_q, lower, upper, max_delta):
    action = np.asarray(action, dtype=float)
    if action.shape not in ((7,), (8,)) or not np.isfinite(action).all():
        raise ValueError("predicted action must have seven arm positions and optional padding")
    q = action[:7].copy()
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("predicted joint target is outside the task/joint envelope")
    if np.max(abs(q - observed_q)) > max_delta:
        raise ValueError("predicted target is too far from measured arm position")
    return q


class SmoothReference:
    """Critically damped target follower, bounded in velocity and acceleration."""
    def __init__(self, q, max_velocity, max_acceleration):
        self.q = np.asarray(q, dtype=float).copy()
        self.dq = np.zeros(7)
        self.max_velocity, self.max_acceleration = max_velocity, max_acceleration

    def step(self, target, dt):
        if not np.isfinite(dt) or dt <= 0 or dt > .1:
            raise RuntimeError("reference control clock stalled")
        accel = np.clip(100 * (target - self.q) - 20 * self.dq,
                        -self.max_acceleration, self.max_acceleration)
        next_dq = np.clip(self.dq + accel * dt, -self.max_velocity, self.max_velocity)
        self.q = self.q + .5 * (self.dq + next_dq) * dt
        self.dq = next_dq
        return self.q.copy(), self.dq.copy()


class RobotServo(threading.Thread):
    def __init__(self, robot, initial_q, args, bounds, control_cpus, control_guard=None):
        super().__init__(daemon=True)
        self.robot, self.args, self.bounds, self.control_cpus = robot, args, bounds, control_cpus
        self.filter = SmoothReference(initial_q, args.max_velocity, args.max_acceleration)
        self.lock = threading.Lock()
        self.target = np.asarray(initial_q).copy()
        self.updated_at = time.monotonic()
        self.abort, self.ready, self.done = threading.Event(), threading.Event(), threading.Event()
        self.error, self.commands = None, []
        self.q_ref, self.dq_ref = self.target.copy(), np.zeros(7)
        self.control_guard = control_guard

    def publish(self, action, observed_q, observation_time):
        target = checked_target(action, observed_q, *self.bounds, self.args.max_target_delta)
        with self.lock:
            self.target, self.updated_at = target, observation_time

    def run(self):
        try:
            from franky import JointImpedanceTracker
            if self.control_cpus:
                os.sched_setaffinity(0, self.control_cpus)
            with JointImpedanceTracker(self.robot, stiffness=REPLAY_STIFFNESS, damping=REPLAY_DAMPING,
                                       lower_joint_limits=Q_MIN, upper_joint_limits=Q_MAX,
                                       period=1 / CONTROL_HZ) as tracker:
                self.updated_at = time.monotonic()
                self.ready.set()
                while not self.abort.is_set():
                    dt = tracker.tick()
                    if dt is None:
                        raise RuntimeError("robot left impedance control")
                    now = time.monotonic()
                    if self.control_guard is not None:self.control_guard()
                    with self.lock:
                        target, updated = self.target.copy(), self.updated_at
                    if now - updated > self.args.watchdog:
                        raise RuntimeError("policy command watchdog expired")
                    q, dq = self.filter.step(target, dt)
                    if np.any(q < self.bounds[0]) or np.any(q > self.bounds[1]):
                        raise RuntimeError("smoothed reference left task envelope")
                    error = np.max(abs(np.asarray(self.robot.state.q) - q))
                    if error > self.args.max_error:
                        raise RuntimeError(f"joint tracking error {error:.4f} rad exceeds {self.args.max_error}")
                    with self.lock:
                        tracker.set_target(q, dq)
                        self.q_ref, self.dq_ref = q, dq
                        self.commands.append((time.monotonic(), q, dq))
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
                raise RuntimeError("robot control thread did not stop")


def save_results(output, records, chunks, commands, result):
    with h5py.File(output / "rollout.h5", "w") as f:
        for key in records[0] if records else []:
            value = np.asarray([r[key] for r in records])
            f.create_dataset(key, data=value, compression="lzf" if value.ndim > 2 else None)
        if chunks:
            f.create_dataset("queries/tick", data=[q[0] for q in chunks])
            f.create_dataset("queries/actions", data=np.stack([q[1] for q in chunks]))
        if commands:
            t0 = result["monotonic_t0"]
            f.create_dataset("control/t", data=[c[0]-t0 for c in commands])
            f.create_dataset("control/q_ref", data=np.stack([c[1] for c in commands]))
            f.create_dataset("control/dq_ref", data=np.stack([c[2] for c in commands]))
        f.attrs["camera_names_json"] = json.dumps(result["camera_names"])
        f.attrs["note"] = "RGB is uint8 in training camera order; targets are absolute arm joint positions in radians"
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def offline(args, policy, output):
    import cv2
    row = args.offline_row
    records, queries = [], []
    player = ChunkPlayer(policy.query_every, args.temporal_agg, args.aggregation_decay)
    with h5py.File(row / "trajectory.h5") as f, ExitStack() as cleanup:
        g = f["traj_0"]; q = g["real/q"][:]; t = g["real/t"][:]
        truth = g["real/target_q"][:]
        caps = {}
        for name in policy.camera_names:
            caps[name] = cv2.VideoCapture(str(row / f"episode_00000000_{name}.mp4"))
            cleanup.callback(caps[name].release)
        for tick in range(min(len(q), args.offline_steps or len(q))):
            images = {}
            for name, cap in caps.items():
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"{name}: cannot read RGB tick {tick}")
                images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = policy.prepare_images(images)
            latency = 0.0
            if player.needs_query(tick):
                since = time.monotonic()
                chunk = policy.predict(q[tick], rgb)
                latency = time.monotonic()-since
                queries.append((tick, chunk))
                player.add(tick, chunk)
            prediction = player.action(tick)
            records.append(dict(t=float(t[tick]), q=q[tick], action=prediction[:7],
                                reference_action=truth[tick],
                                loader_previous_action=truth[max(0,tick-1)], inference_s=latency, rgb=rgb))
    pred = np.stack([r["action"] for r in records])
    target = np.stack([r["reference_action"] for r in records])
    previous = np.stack([r["loader_previous_action"] for r in records])
    result = dict(mode="offline_teacher_forced", success=None, checkpoint=str(args.ckpt_dir / args.ckpt_name),
                  camera_names=policy.camera_names, steps=len(records), queries=len(queries),
                  arm_mae_rad=float(np.mean(abs(pred-target))), max_arm_error_rad=float(abs(pred-target).max()),
                  previous_action_mae_rad=float(np.mean(abs(pred-previous))),
                  query_latency_s=[r["inference_s"] for r in records if r["inference_s"]],
                  note="Recorded observations; no hardware connection and no claim of closed-loop success. "
                       "Current ACT training loader shifts real action labels by one sample.")
    save_results(output, records, queries, [], result)
    print(json.dumps({k:v for k,v in result.items() if k != "query_latency_s"}, indent=2))


def live(args, policy, output, control_cpus, context=None):
    from franky import JointMotion, Robot

    camera_args = argparse.Namespace(**vars(args))
    if not base.camera_specs(camera_args):
        raise ValueError("configure cameras")
    configured = dict(base.camera_specs(camera_args))
    if set(configured) != set(policy.camera_names):
        raise ValueError(f"configured camera names must match trained names {policy.camera_names}")
    bounds = (np.maximum(np.minimum(args.start, args.goal)-args.task_margin, np.asarray(Q_MIN)+.05),
              np.minimum(np.maximum(args.start, args.goal)+args.task_margin, np.asarray(Q_MAX)-.05))
    records, queries, servo = [], [], None
    failure = None
    result = dict(mode="real_robot", checkpoint=str(args.ckpt_dir/args.ckpt_name),
                  camera_names=policy.camera_names, start_q=args.start, goal_q=args.goal,
                  success=False, robot_connected=False, termination_phase='sensor_preflight' if context is not None else 'camera_setup',
                  monotonic_t0=time.monotonic())
    try:
        with ExitStack() as cleanup:
            if context is not None:context.setup(cleanup)
            result['termination_phase']='camera_setup'
            cameras = base.open_cameras(camera_args)
            for cam in cameras:
                cleanup.callback(cam.close)
            cams = {cam.name:cam for cam in cameras}
            result['termination_phase']='robot_connection'
            robot = Robot(args.host)
            result['robot_connected']=True
            robot.recover_from_errors()
            robot.relative_dynamics_factor = args.dynamics
            if not args.no_reset:
                result['termination_phase']='reset_to_start'
                print("Moving to training start A, outside rollout.")
                robot.move(JointMotion(args.start))
                time.sleep(.5)
            result['termination_phase']='start_pose_check'
            initial = base.read_robot_state(robot)["q"]
            if np.max(abs(initial-np.asarray(args.start))) > args.goal_tolerance:
                advice=("remove the obstacle, run 'python eval_pc_act_real.py --reset-only', then start the next trial. "
                        "For a clear reset path, --reset-to-start can reset before a PC-ACT run.") if context is not None else (
                        "check the reset motion and start pose; omit --no-reset to reset before evaluation.")
                raise RuntimeError("arm is not at training start A; "+advice)
            # Prepare the first chunk at A before starting torque control. All
            # subsequent inference runs in one worker, off the sampling thread.
            bootstrap_images = {}
            result['termination_phase']='model_bootstrap'
            for name in policy.camera_names:
                frame, _ = cams[name].latest()
                if frame is None or time.monotonic()-frame[4]>args.camera_timeout:
                    raise RuntimeError(f"{name}: camera frame stale or unavailable")
                bootstrap_images[name] = frame[0]
            initial_context = context.sample() if context is not None else None
            initial_chunk, initial_latency = predict_timed(policy, initial, policy.prepare_images(bootstrap_images),
                context.policy_input(initial_context) if context is not None else None)
            checked_target(initial_chunk[0], initial, *bounds, args.max_target_delta)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-inference")
            cleanup.callback(executor.shutdown, wait=True, cancel_futures=True)
            servo = RobotServo(robot, initial, args, bounds, control_cpus,
                               context.control_guard if context is not None else None)
            cleanup.callback(servo.stop)
            result['termination_phase']='control_startup'
            servo.start()
            if not servo.ready.wait(3) or servo.error is not None:
                raise RuntimeError(f"could not start impedance control: {servo.error}")
            player = ChunkPlayer(policy.query_every, args.temporal_agg, args.aggregation_decay)
            player.add(0, initial_chunk)
            queries.append((0, initial_chunk))
            result["query_latency_s"] = [initial_latency]
            pending = None
            t0 = time.monotonic();result["monotonic_t0"] = t0
            result['termination_phase']='rollout'
            for tick in range(int(np.ceil(args.duration*base.RATE_HZ))):
                remaining=t0+tick/base.RATE_HZ-time.monotonic()
                if remaining>0:time.sleep(remaining)
                observed_at=time.monotonic()
                sensor_context = context.sample() if context is not None else None
                if observed_at-(t0+tick/base.RATE_HZ)>args.watchdog:
                    raise RuntimeError("policy loop fell behind its 15 Hz clock")
                if servo.error is not None or servo.done.is_set():
                    raise RuntimeError(f"robot control stopped: {servo.error}")
                with servo.lock:
                    state=base.read_robot_state(robot)
                    qref,dqref=servo.q_ref.copy(),servo.dq_ref.copy()
                    if context is not None:sensor_context['robot_fields']=context.robot_fields(robot)
                images,ages,numbers={},[],[]
                for name in policy.camera_names:
                    frame,_=cams[name].latest()
                    if frame is None:raise RuntimeError(f"{name}: no frame")
                    rgb,_,number,_,arrived=frame
                    age=time.monotonic()-arrived
                    if age>args.camera_timeout:raise RuntimeError(f"{name}: camera frame stale ({age:.3f}s)")
                    images[name]=rgb;ages.append(age);numbers.append(number)
                rgb=policy.prepare_images(images);latency=0.0
                if pending is not None:
                    query_tick, query_at, future = pending
                    if future.done():
                        chunk, latency = future.result()
                        if time.monotonic()-query_at>args.watchdog:
                            raise RuntimeError("background inference exceeded latency watchdog")
                        player.add(query_tick,chunk);queries.append((query_tick,chunk))
                        result["query_latency_s"].append(latency)
                        pending = None
                    elif time.monotonic()-query_at>args.watchdog:
                        raise RuntimeError("background inference stalled")
                due = 1 if args.temporal_agg else player.next_query_tick(args.prefetch_steps)
                if pending is None and tick >= due:
                    sensor_input = context.policy_input(sensor_context) if context is not None else None
                    pending = (tick, time.monotonic(), executor.submit(predict_timed,policy,state["q"].copy(),rgb.copy(),sensor_input))
                if time.monotonic()-observed_at>args.watchdog:
                    raise RuntimeError("observation/inference latency exceeded watchdog")
                action=player.action(tick)
                servo.publish(action,state["q"],observed_at)
                records.append(dict(t=observed_at-t0,q=state["q"],dq=state["dq"],q_ref=qref,dq_ref=dqref,
                                    action=action[:7],inference_s=latency,rgb=rgb,camera_age_s=np.asarray(ages),
                                    camera_frame_number=np.asarray(numbers)))
                if context is not None:records[-1].update(context.record_fields(sensor_context))
                if tick%15==0:print(f"t={observed_at-t0:.1f}s goal error={np.max(abs(state['q']-args.goal)):.4f} rad")
            final=records[-1]
            # Result is computed while holding position, before controller shutdown.
            result["final_goal_error_rad"]=float(np.max(abs(final["q"]-args.goal)))
            result["final_max_speed_rad_s"]=float(np.max(abs(final["dq"])))
            tail=[r for r in records if r["t"]>=final["t"]-args.success_hold]
            result["success"]=bool(len(tail)>=max(2,int(args.success_hold*base.RATE_HZ)) and
                all(np.max(abs(r["q"]-args.goal))<=args.goal_tolerance and np.max(abs(r["dq"]))<=args.settle_velocity for r in tail))
            result['termination_phase']='completed'
    except BaseException as exc:
        failure=exc
        result["success"]=False
        raise
    finally:
        if servo is not None and servo.error is not None:
            failure=failure or servo.error;result["success"]=False
        result.update(error="" if failure is None else repr(failure),steps=len(records),
                      queries=len(queries),query_every=policy.query_every,temporal_agg=args.temporal_agg,
                      controller="100 Hz acceleration/velocity-limited joint impedance reference follower")
        if context is not None:result.update(context.result_fields())
        save_results(output,records,queries,servo.commands if servo else [],result)
        if context is not None:context.save_events(output,result["monotonic_t0"])
        print(f"saved {output}; success={result['success']}")
    if failure is not None:raise failure


def parse_args(argv=None,configure=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir",type=Path,default=DEFAULT_RUN)
    p.add_argument("--ckpt-name",default="policy_best.ckpt")
    p.add_argument("--policy-config",type=Path,help="JSON policy_config for checkpoints without saved metadata")
    p.add_argument("--data-dir",type=Path,help="converted training dataset directory")
    p.add_argument("--camera-names",nargs="+",help="exact camera order used during training")
    p.add_argument("--image-height",type=int)
    p.add_argument("--image-width",type=int)
    p.add_argument("--nheads",type=int,help="only needed if training metadata is missing")
    p.add_argument("--device",help="default: cuda if available, otherwise cpu")
    p.add_argument("--torch-threads",type=int,default=2)
    p.add_argument("--query-every",type=int,help="default: execute the checkpoint's full chunk")
    p.add_argument("--temporal-agg",action="store_true",help="query every tick and aggregate overlapping chunks")
    p.add_argument("--prefetch-steps",type=int,default=8,help="start background inference this many ticks before chunk expiry")
    p.add_argument("--aggregation-decay",type=float,default=.01,help="positive values favor newer predictions")
    p.add_argument("--offline-row",type=Path,help="raw recorded row for inference only; no hardware connection")
    p.add_argument("--check-model",action="store_true",help="load model and run one dummy inference; no cameras or robot")
    p.add_argument("--offline-steps",type=int,help="limit recorded observations used in offline mode")
    p.add_argument("--host",default="172.16.0.2")
    p.add_argument("--start",type=float,nargs=7,default=START)
    p.add_argument("--goal",type=float,nargs=7,default=GOAL)
    p.add_argument("--duration",type=float,default=10.1)
    p.add_argument("--no-reset",action="store_true",help="require arm already at A instead of moving to A")
    p.add_argument("--dynamics",type=float,default=.05,help="reset JointMotion dynamics")
    p.add_argument("--max-velocity",type=float,default=.15,help="reference rad/s")
    p.add_argument("--max-acceleration",type=float,default=.5,help="reference rad/s^2")
    p.add_argument("--max-error",type=float,default=.15,help="maximum measured joint tracking error rad")
    p.add_argument("--max-target-delta",type=float,default=.12,help="maximum target minus observed joint position rad")
    p.add_argument("--task-margin",type=float,default=.08,help="joint envelope beyond the A/B range rad")
    p.add_argument("--goal-tolerance",type=float,default=.02)
    p.add_argument("--settle-velocity",type=float,default=.01)
    p.add_argument("--success-hold",type=float,default=.5)
    p.add_argument("--watchdog",type=float,default=.5,help="maximum command or observation delay seconds")
    p.add_argument("--camera-timeout",type=float,default=.25)
    p.add_argument("--control-cpu",type=int)
    p.add_argument("--wrist-serial")
    p.add_argument("--exo-serial")
    p.add_argument("--camera",nargs=2,action="append",default=[],metavar=("NAME","SERIAL"))
    p.add_argument("--no-camera-reset",action="store_true")
    p.add_argument("--cam-width",type=int,default=640)
    p.add_argument("--cam-height",type=int,default=480)
    p.add_argument("--cam-fps",type=int,default=30)
    p.add_argument("--output",type=Path,help="new rollout output directory")
    if configure is not None:configure(p)
    args=p.parse_args(argv)
    for key in ("duration","dynamics","max_velocity","max_acceleration","max_error","max_target_delta",
                "task_margin","goal_tolerance","settle_velocity","success_hold","watchdog","camera_timeout"):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0:p.error(f"--{key.replace('_','-')} must be positive and finite")
    if args.dynamics>1:p.error("--dynamics must be <=1")
    if args.success_hold>=args.duration:p.error("success-hold must be shorter than duration")
    if args.torch_threads<1:p.error("torch-threads must be positive")
    if args.prefetch_steps<1:p.error("prefetch-steps must be positive")
    if args.query_every is not None and args.query_every<1:p.error("query-every must be positive")
    if args.offline_steps is not None and args.offline_steps<1:p.error("offline-steps must be positive")
    if not np.isfinite(args.aggregation_decay) or args.aggregation_decay<0:p.error("aggregation-decay must be nonnegative")
    for q in (args.start,args.goal):
        if not np.isfinite(q).all() or np.any(np.asarray(q)<np.asarray(Q_MIN)+.05) or np.any(np.asarray(q)>np.asarray(Q_MAX)-.05):
            p.error("start/goal must be finite and within FR3 joint limits with 0.05 rad margin")
    if not args.offline_row and not args.check_model and not getattr(args,'sensor_check',False) and not getattr(args,'reset_only',False):
        try:base.camera_specs(args)
        except ValueError as exc:p.error(str(exc))
    return args


def main(argv=None):
    args=parse_args(argv)
    output=args.output or ROOT/"eval_rollouts"/datetime.now(timezone.utc).strftime("act_%Y%m%dT%H%M%S%fZ")
    if output.exists():raise FileExistsError(f"output directory already exists: {output}")
    # Reserve control CPU before Torch workers and camera threads are created.
    affinity=os.sched_getaffinity(0)
    try:
        control_cpus=None if args.offline_row or args.check_model else base.reserve_control_cpus(args.control_cpu)
        policy=ACTInference(args)
        dummy=np.zeros((len(policy.camera_names),policy.height,policy.width,3),dtype=np.uint8)
        prediction=policy.predict(np.asarray(args.start),dummy)  # initialize kernels before control starts
        if args.check_model:
            print(json.dumps(dict(checkpoint=str(args.ckpt_dir/args.ckpt_name),
                                  camera_names=policy.camera_names, image_size=[policy.height,policy.width],
                                  action_shape=list(prediction.shape), finite=bool(np.isfinite(prediction).all()),
                                  device=policy.device, hardware_connected=False),indent=2))
            return
        output.mkdir(parents=True,exist_ok=False)
        config=dict(vars(args),resolved_camera_names=policy.camera_names,policy_config=policy.config,
                    image_height=policy.height,image_width=policy.width,resolved_device=policy.device)
        (output/"config.json").write_text(json.dumps(config,default=str,indent=2)+"\n")
        if args.offline_row:offline(args,policy,output)
        else:live(args,policy,output,control_cpus)
    finally:
        os.sched_setaffinity(0,affinity)


if __name__=="__main__":main()
