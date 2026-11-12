"""Evaluate an ACT or PC-ACT block-pick policy on the FR3, gripper included.

The policies come from the prox_learning sim recipe (submodules/act/imitate_episodes.py through
scripts/exp/train_real_pick.sh, task real_pick_20261112). Export a run into a portable bundle
first, then check it offline, rehearse it in simulation, and only then run it on the arm:

    python model.py export pick --source <imitate_episodes run dir> --destination models/pick_act
    python model.py eval pick --model models/pick_act --check-model
    python model.py eval pick --model models/pick_act --offline recordings/pick_20261112T022139Z
    python model.py eval pick --model models/pick_act --sim
    python model.py eval pick --model models/pick_act --wrist-serial <s> --exo-serial <s> --trials 3

Observations are built exactly as training saw them: the 7 measured joints plus two finger
positions (gripper width / 2 each), both cameras resized to the training size with INTER_AREA,
and for PC-ACT the newest packet of every skin sensor (block 1, metres, no return = 4 m) kept as
a causal 8-tick 15 Hz history and encoded with encode_pooled_history. The policy runs at 15 Hz
and executes each 50-step chunk before querying again (the sim protocol: exec horizon = chunk,
no temporal aggregation; --exec-horizon / --temporal-agg change it). Its arm targets are
interpolated to 100 Hz for franky's joint impedance tracker and rate limited. A gripper output
above 0.5 closes the hand with the collection grasp (20 N, any width), below 0.5 opens it.

Every trial starts at HOME with the gripper open, like the recorded episodes. The arm stops
(short damping stop) when the target leaves the joint limits, the hand would go more than
--floor-margin below the saved grasp height or leave the workspace box, a joint lags its
reference by --max-error, the policy stalls, or --duration runs out. A trial ends early once
the hand holds the block --lift-success above the grasp height for --hold-success s. Success =
holding the block (width above --min-width) and lifted. Then, not part of the policy, the
script puts the block back on its spot the way data.py pick does (or, with nothing in the
hand, opens, rises and goes home), so the next trial finds it there.

Each trial writes evaluations/pick_<model>_<utc>/trial_NNNN/{rollout.h5,result.json}.
"""

import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import threading
import time

import h5py
import numpy as np

from . import pick_block as pick
from . import record_dataset as base
from .kinesthetic import Q_MAX, Q_MIN, REPLAY_DAMPING, REPLAY_STIFFNESS

ROOT = Path(__file__).resolve().parents[1]
POLICY_HZ = 15.0
CONTROL_HZ = pick.CONTROL_HZ
NO_RETURN_M = 4.0
MANIFEST = "pick_policy.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ---- export ---------------------------------------------------------------------------------
def export_bundle(source, destination, ckpt_name="policy_best.ckpt", data_dir=None):
    """imitate_episodes run dir -> portable bundle: weights, npz stats and a manifest."""
    import shutil

    source, destination = Path(source), Path(destination)
    policy = source / ckpt_name
    if not policy.is_file():
        raise FileNotFoundError(f"{policy} missing (training not finished?)")
    with (source / "dataset_stats.pkl").open("rb") as f:
        stats = pickle.load(f)
    stats = {k: np.asarray(stats[k], dtype=np.float32) for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")}
    prox = None
    files = {"policy": ckpt_name}
    if (source / "prox_config.json").exists():
        prox = json.loads((source / "prox_config.json").read_text())
        if not prox.get("finetune_prox_encoder") or prox.get("prox_policy_tap") != "readout":
            raise ValueError("only the finetuned readout PC-ACT (PACT_READOUT) is supported")
        encoder = {"policy_best.ckpt": "prox_encoder_best.pt", "policy_last.ckpt": "prox_encoder.pt"}.get(
            ckpt_name, ckpt_name.replace("policy_", "prox_encoder_").replace(".ckpt", ".pt"))
        pairs = json.loads((source / "checkpoint_pairs.json").read_text())
        pair = pairs.get(ckpt_name)
        if pair is None or pair["encoder"] != encoder:
            raise ValueError(f"no recorded policy/encoder pair for {ckpt_name}")
        if sha256(policy) != pair["policy_sha256"] or sha256(source / encoder) != pair["encoder_sha256"]:
            raise ValueError("policy/encoder pair hash mismatch: the run is still writing or was modified")
        files["encoder"] = encoder
    meta = {}
    if data_dir is not None:
        meta = json.loads((Path(data_dir) / "convert_meta.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    for name in files.values():
        shutil.copy2(source / name, destination / name)
    np.savez(destination / "dataset_stats.npz", **stats)
    if prox is not None:
        (destination / "prox_config.json").write_text(json.dumps(prox, indent=2) + "\n")
    manifest = dict(
        kind="pc-act" if prox else "act", source=str(source.resolve()), checkpoint=ckpt_name,
        camera_names=meta.get("camera_names", ["exo_camera_1", "wrist_camera"]),
        image_h=int(meta.get("image_h", 240)), image_w=int(meta.get("image_w", 320)),
        nheads=8, enc_layers=4, dec_layers=7, backbone="resnet18",
        files={k: dict(name=v, sha256=sha256(destination / v)) for k, v in files.items()},
        recipe="submodules/act/imitate_episodes.py via scripts/exp/train_real_pick.sh (sim hidden-hazard recipe)",
        exported_utc=datetime.now(timezone.utc).isoformat())
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest['kind']} bundle -> {destination}")
    return destination


# ---- policy ---------------------------------------------------------------------------------
class PickPolicy:
    """Inference-only ACT / PC-ACT on an exported bundle; predict() returns an unnormalized chunk."""

    def __init__(self, model_dir, device=None):
        import torch

        from act_model import build_model

        self.torch = torch
        self.dir = Path(model_dir)
        self.manifest = json.loads((self.dir / MANIFEST).read_text())
        for key, entry in self.manifest["files"].items():
            path = self.dir / entry["name"]
            with path.open("rb") as f:
                if f.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
                    raise ValueError(f"{path} is a Git LFS pointer; run git lfs pull")
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"{path}: checksum differs from the exported bundle")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with np.load(self.dir / "dataset_stats.npz") as f:
            self.stats = {k: f[k].astype(np.float32) for k in f.files}
        state = torch.load(self.dir / self.manifest["files"]["policy"]["name"], map_location="cpu", weights_only=True)
        self.state_dim = int(state["model.input_proj_robot_state.weight"].shape[1])
        self.action_dim = int(state["model.action_head.weight"].shape[0])
        self.chunk = int(state["model.query_embed.weight"].shape[0])
        if (self.state_dim, self.action_dim) != (9, 8):
            raise ValueError(f"expected a 9-state / 8-action pick policy, got {self.state_dim}/{self.action_dim}")
        self.cameras = list(self.manifest["camera_names"])
        self.height, self.width = self.manifest["image_h"], self.manifest["image_w"]
        self.kind = self.manifest["kind"]
        config = dict(camera_names=self.cameras, num_queries=self.chunk, state_dim=9, action_dim=8,
                      hidden_dim=int(state["model.query_embed.weight"].shape[1]),
                      dim_feedforward=int(state["model.transformer.encoder.layers.0.linear1.weight"].shape[0]),
                      enc_layers=self.manifest["enc_layers"], dec_layers=self.manifest["dec_layers"],
                      nheads=self.manifest["nheads"], backbone=self.manifest["backbone"], lr_backbone=1e-5)
        self.encoder = None
        if self.kind == "pc-act":
            from act_model.surface_geometry import SurfaceGeometryEncoder

            prox = json.loads((self.dir / "prox_config.json").read_text())
            config.update(n_proximity_sensors=int(prox["n_proximity_sensors"]),
                          prox_tokens_per_sensor=int(prox["prox_tokens_per_sensor"]),
                          prox_feat_dim=int(prox["prox_feat_dim"]))
            self.sensor_order = list(prox["sensor_order"])
            self.encoder = SurfaceGeometryEncoder(kind="embedding", checkpoint=self.dir / self.manifest["files"]["encoder"]["name"],
                                                  device=self.device, frozen=True, policy_tap="readout").eval()
        model = build_model(config)
        model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 1, 3, 1, 1)

    def prepare_images(self, images):
        import cv2

        out = []
        for name in self.cameras:
            rgb = np.asarray(images[name])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"{name}: expected a uint8 RGB image")
            if rgb.shape[:2] != (self.height, self.width):
                rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
            out.append(rgb)
        return np.stack(out)

    def predict(self, qpos, images, skin_history=None):
        """qpos (9,), images {camera: RGB} or prepared (cams, H, W, 3), skin (8, 40, 8, 8) m -> (chunk, 8)."""
        torch = self.torch
        rgb = images if isinstance(images, np.ndarray) else self.prepare_images(images)
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (9,) or not np.isfinite(qpos).all():
            raise ValueError("qpos must be 9 finite values")
        state = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        with torch.inference_mode():
            q_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            img = torch.as_tensor(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)), device=self.device).float().unsqueeze(0) / 255
            features = None
            if self.encoder is not None:
                skin = np.asarray(skin_history, dtype=np.float32)
                if skin.shape != (8, 40, 8, 8) or not np.isfinite(skin).all():
                    raise ValueError("PC-ACT needs a finite (8, 40, 8, 8) skin history in metres")
                features = self.encoder.encode_pooled_history(torch.as_tensor(skin, device=self.device).unsqueeze(0))
            action, _, _ = self.model(q_t, (img - self.mean) / self.std, None, proximity_positions=features)
        action = action[0].float().cpu().numpy()
        if action.shape != (self.chunk, 8) or not np.isfinite(action).all():
            raise ValueError("policy produced an invalid action chunk")
        return action * self.stats["action_std"] + self.stats["action_mean"]


# ---- observations ---------------------------------------------------------------------------
def qpos_vector(q, width):
    """7 measured joints + both fingers at width / 2, as the recorder wrote obs/agent/qpos."""
    finger = float(width) / 2.0 if np.isfinite(width) else 0.0
    return np.concatenate([np.asarray(q, dtype=np.float32).reshape(7), [finger, finger]]).astype(np.float32)


def sensor_index(order):
    """Position of each policy sensor (sim name) in base.SENSORS order."""
    from .sensor_layout import sim_name

    names = [sim_name(link, ident) for link, ident in base.SENSORS]
    return [names.index(name) for name in order]


def skin_frame(depth_mm, index):
    """(40, 128) uint16 mm packets in base.SENSORS order -> (40, 8, 8) m in policy order (block 1)."""
    meters = np.asarray(depth_mm, dtype=np.float32)[index, :64].reshape(len(index), 8, 8) / 1000.0
    meters[(meters <= 0) | (meters >= NO_RETURN_M)] = NO_RETURN_M
    return meters


class SkinHistory:
    """Last 8 frames at 15 Hz, left-padded with the oldest, as the training loader (raw_causal)."""

    def __init__(self):
        self.frames = deque(maxlen=8)

    def append(self, frame):
        self.frames.append(np.asarray(frame, dtype=np.float32))
        frames = [self.frames[0]] * (8 - len(self.frames)) + list(self.frames)
        return np.stack(frames)


def live_packets(listener):
    """Newest packet of every sensor; a silent sensor reads no return, as in the recordings."""
    depth = np.full((len(base.SENSORS), 128), int(NO_RETURN_M * 1000), np.uint16)
    if listener is not None:
        packets = listener.snapshot()
        for i, key in enumerate(base.SENSORS):
            if key in packets:
                depth[i] = packets[key][0]
    return depth


# ---- offline ---------------------------------------------------------------------------------
def row_observations(row, policy):
    """Rebuild the policy inputs of a recorded row from its raw files (no converter)."""
    import cv2

    row = Path(row)
    with h5py.File(row / "trajectory.h5") as f:
        t = f["traj_0"]
        q = t["real/q"][:]
        width = t["real/gripper_width"][:]
        depth = t["real/depth_mm"][:]                           # (T, 40, 2, 8, 8)
        target = t["real/target_q"][:]
        command = t["real/gripper_command"][:]
    T = len(q)
    frames = {}
    for name in policy.cameras:
        cap, out = cv2.VideoCapture(str(row / f"episode_00000000_{name}.mp4")), []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(cv2.resize(bgr, (policy.width, policy.height), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB))
        if len(out) != T:
            raise ValueError(f"{row}: {name} has {len(out)} frames for {T} ticks")
        frames[name] = np.asarray(out)
    images = np.stack([frames[n] for n in policy.cameras], axis=1)
    skin = None
    if policy.encoder is not None:
        index = sensor_index(policy.sensor_order)
        skin = np.stack([skin_frame(depth[i].reshape(len(base.SENSORS), 128), index) for i in range(T)])
    qpos = np.stack([qpos_vector(q[i], width[i]) for i in range(T)])
    actions = np.concatenate([target, command[:, None]], axis=1).astype(np.float32)
    return qpos, images, skin, actions


def offline(policy, rows, exec_horizon):
    """Open-loop check on recorded rows: predicted vs recorded actions, arm and gripper."""
    results = []
    for row in rows:
        qpos, images, skin, actions = row_observations(row, policy)
        T = len(qpos)
        errors, first, grip_agree, grip_n = [], [], 0, 0
        executed = np.zeros_like(actions)
        for t in range(0, T, exec_horizon):
            # Same window as the training loader: last 8 ticks, left-padded with the first.
            hist = None if skin is None else skin[[max(0, k) for k in range(t - 7, t + 1)]]
            chunk = policy.predict(qpos[t], images[t], hist)
            n = min(exec_horizon, T - t, policy.chunk)
            executed[t:t + n] = chunk[:n]
            ref = actions[t:t + policy.chunk]
            errors.append(np.abs(chunk[:len(ref), :7] - ref[:, :7]).mean())
            first.append(np.abs(chunk[0, :7] - actions[t, :7]).max())
        grip_agree = int(((executed[:, 7] > 0.5) == (actions[:, 7] > 0.5)).sum())
        grip_n = T
        close_pred = np.flatnonzero(executed[:, 7] > 0.5)
        close_true = np.flatnonzero(actions[:, 7] > 0.5)
        result = dict(row=str(row), ticks=T, chunk_arm_mae_rad=float(np.mean(errors)),
                      first_step_max_err_rad=float(np.max(first)),
                      executed_arm_max_err_rad=float(np.abs(executed[:, :7] - actions[:, :7]).max()),
                      gripper_agreement=grip_agree / grip_n,
                      close_tick_predicted=int(close_pred[0]) if len(close_pred) else None,
                      close_tick_recorded=int(close_true[0]) if len(close_true) else None)
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


# ---- online ----------------------------------------------------------------------------------
class Envelope:
    """Joint limits, a floor under the grasp height and a workspace box for the hand."""

    def __init__(self, saved, args):
        self.f_t_ee = np.linalg.inv(pick.flange(np.asarray(saved["grasp_q"]))) @ np.asarray(saved["grasp_pose"])
        grasp = np.asarray(saved["grasp_pose"])[:3, 3]
        home = (pick.flange(np.asarray(pick.HOME)) @ self.f_t_ee)[:3, 3]
        self.floor = grasp[2] - args.floor_margin
        self.lo = np.minimum(grasp, home) - args.box_margin
        self.hi = np.maximum(grasp, home) + args.box_margin
        self.q_lo, self.q_hi = np.asarray(Q_MIN) + 0.05, np.asarray(Q_MAX) - 0.05
        self.grasp_z = grasp[2]

    def hand(self, q):
        return (pick.flange(q) @ self.f_t_ee)[:3, 3]

    def check(self, q):
        if np.any(q < self.q_lo) or np.any(q > self.q_hi):
            j = int(np.argmax(np.maximum(self.q_lo - q, q - self.q_hi)))
            raise RuntimeError(f"target joint {j + 1} within 0.05 rad of its limit")
        p = self.hand(q)
        if p[2] < self.floor:
            raise RuntimeError(f"target hand height {p[2]:.3f} m is below the floor {self.floor:.3f} m")
        if np.any(p < self.lo) or np.any(p > self.hi):
            raise RuntimeError(f"target hand position {np.round(p, 3).tolist()} m leaves the workspace box")


class Servo(threading.Thread):
    """100 Hz: interpolate the newest chunk, rate limit, check, stream; then run the put-back."""

    def __init__(self, robot, gripper, envelope, plan, args, control_cpus):
        super().__init__(daemon=True)
        self.robot, self.gripper, self.envelope, self.plan, self.args = robot, gripper, envelope, plan, args
        self.control_cpus = control_cpus
        self.lock = threading.Lock()
        self.chunk = None                    # (t0, (N, 7) arm targets at 15 Hz from t0)
        self.q_ref = np.asarray(pick.HOME, dtype=np.float64).copy()
        self.max_error_seen = 0.0
        self.error = None
        self.policy_done, self.finish, self.done, self.ready = (threading.Event() for _ in range(4))
        self.holding = False
        self.put_back_allowed = not args.no_put_back
        self.commands = []

    def publish(self, t0, arm):
        with self.lock:
            self.chunk = (t0, np.asarray(arm, dtype=np.float64))

    def hold(self):
        """Stop following the policy and hold the current reference."""
        with self.lock:
            self.chunk = None

    def target(self, now):
        with self.lock:
            chunk = self.chunk
        if chunk is None:
            return self.q_ref
        t0, arm = chunk
        s = (now - t0) * POLICY_HZ
        if s > len(arm) - 1 + POLICY_HZ * self.args.watchdog:
            raise RuntimeError("policy stalled: no new action chunk")
        s = min(max(s, 0.0), len(arm) - 1)
        i = int(np.floor(s))
        j = min(i + 1, len(arm) - 1)
        return arm[i] + (s - i) * (arm[j] - arm[i])

    def run(self):
        from franky import JointImpedanceTracker

        tracker = None
        try:
            if self.control_cpus:
                os.sched_setaffinity(0, self.control_cpus)
            offset = np.abs(np.asarray(self.robot.state.q) - pick.HOME)
            if offset.max() > pick.START_TOLERANCE:
                raise RuntimeError(f"arm is not at HOME (joint {int(offset.argmax()) + 1} off by {offset.max():.3f} rad)")
            tracker = JointImpedanceTracker(self.robot, stiffness=REPLAY_STIFFNESS, damping=REPLAY_DAMPING,
                                            lower_joint_limits=Q_MIN, upper_joint_limits=Q_MAX, period=1 / CONTROL_HZ)
            self.ready.set()
            step = self.args.max_velocity / CONTROL_HZ
            while not self.policy_done.is_set():
                if tracker.tick() is None:
                    raise RuntimeError("robot left impedance control")
                goal = self.target(time.monotonic())
                q = self.q_ref + np.clip(goal - self.q_ref, -step, step)
                self.envelope.check(q)
                measured = np.asarray(self.robot.state.q)
                if np.abs(goal - measured).max() > self.args.max_target_delta:
                    raise RuntimeError(f"policy target {np.abs(goal - measured).max():.3f} rad from the measured arm")
                tracker.set_target(q, (q - self.q_ref) * CONTROL_HZ)
                self.q_ref = q
                error = float(np.abs(measured - q).max())
                self.max_error_seen = max(self.max_error_seen, error)
                if error > self.args.max_error:
                    raise RuntimeError(f"tracking error {error:.3f} rad > {self.args.max_error}")
                self.commands.append((time.monotonic(), q.copy()))
            # Put-back (not part of the policy): same segments as data.py pick, starting where the arm is.
            if self.put_back_allowed:
                self.put_back(tracker)
        except BaseException as exc:
            self.error = exc
        finally:
            if tracker is not None:
                try:
                    pick.stop_tracker(tracker)
                except Exception as exc:
                    if self.error is None:
                        self.error = exc
            self.ready.set()
            self.done.set()

    def put_back(self, tracker):
        args, plan, env = self.args, self.plan, self.envelope
        runner = pick.PickRunner(self.robot, self.gripper, plan, args)
        runner.reference = self.q_ref.copy()
        lift_end = dict(plan.pick)["lift"][-1]
        if self.holding:
            if np.abs(lift_end - self.q_ref).max() > args.max_align:
                raise RuntimeError(f"arm is {np.abs(lift_end - self.q_ref).max():.3f} rad from the lifted pose; "
                                   "not putting the block back automatically")
            segments = [("align", pick.joint_segment(self.q_ref, lift_end, args.max_velocity))] + plan.put_back
            print("putting the block back (not part of the policy)", flush=True)
        else:
            pose = pick.flange(self.q_ref) @ env.f_t_ee
            up = pick.shifted(pose, dz=max(0.0, env.grasp_z + max(args.approach, args.lift) - pose[2, 3]))
            segments = [("open", None)]
            if up[2, 3] - pose[2, 3] > 0.005:
                segments.append(("rise", pick.line_segment(self.q_ref, pose, up, env.f_t_ee, args.speed)))
            start = segments[-1][1][-1] if segments[-1][1] is not None else self.q_ref
            segments.append(("home", pick.joint_segment(start, pick.HOME, args.max_velocity)))
            print("nothing in the hand: open, rise, home (not part of the policy)", flush=True)
        for name, path in segments:
            if path is not None and name != "home":       # the home move leaves the box on purpose
                for q in path:
                    env.check(q)
        runner.run_segments(tracker, segments, recorded=False)
        self.max_error_seen = max(self.max_error_seen, runner.max_seen_error)


class GripperCommand:
    """Close above 0.5, open below, never two commands in flight; the collection grasp."""

    def __init__(self, gripper, args):
        self.gripper, self.args = gripper, args
        self.closed, self.pending, self.events = False, None, []

    def update(self, value, now):
        if self.pending is not None and not self.pending.wait(0):
            return
        if self.pending is not None:
            try:
                ok = bool(self.pending.get())
            except Exception as exc:
                ok = False
                print(f"gripper: {exc}", flush=True)
            self.events[-1]["ok"] = ok
            self.pending = None
        want = value > 0.5
        if want and not self.closed:
            self.pending = self.gripper.grasp_async(0.0, self.args.gripper_speed, self.args.grasp_force,
                                                    epsilon_inner=0.005, epsilon_outer=1.0)
            self.closed = True
            self.events.append(dict(command="close", t=now))
        elif not want and self.closed:
            self.pending = self.gripper.open_async(self.args.gripper_speed)
            self.closed = False
            self.events.append(dict(command="open", t=now))

    def wait(self, timeout=10.0):
        if self.pending is not None:
            deadline = time.monotonic() + timeout
            while not self.pending.wait(0):
                if time.monotonic() > deadline:
                    raise RuntimeError("gripper command did not finish")
                time.sleep(0.01)
            try:
                self.events[-1]["ok"] = bool(self.pending.get())
            except Exception:
                self.events[-1]["ok"] = False
            self.pending = None


def run_trial(robot, gripper_reader, cameras, listener, policy, plan, envelope, args, out, control_cpus):
    servo = Servo(robot, gripper_reader.gripper, envelope, plan, args, control_cpus)
    grip = GripperCommand(gripper_reader.gripper, args)
    history = SkinHistory()
    index = sensor_index(policy.sensor_order) if policy.encoder is not None else None
    records, queries = [], []
    chunk, chunk_tick, ensemble = None, -10 ** 9, []
    failure, reason, held_since = None, "duration", None
    servo.start()
    servo.ready.wait(10)
    t0 = time.monotonic()
    try:
        if servo.error is not None:
            raise RuntimeError(f"servo did not start: {servo.error}")
        tick = 0
        while True:
            now = time.monotonic()
            if servo.done.is_set():
                raise RuntimeError(f"servo stopped: {servo.error}")
            state = base.read_robot_state(robot)
            width = float(gripper_reader.width)
            images = {}
            for cam in cameras:
                frame, _ = cam.latest()
                if frame is None:
                    raise RuntimeError(f"{cam.name}: no frame")
                images[cam.name] = frame[0]
            qpos = qpos_vector(state["q"], width)
            depth = live_packets(listener)
            hist = history.append(skin_frame(depth, index)) if index is not None else None
            if args.temporal_agg or tick - chunk_tick >= args.exec_horizon or chunk is None:
                prepared = policy.prepare_images(images)
                since = time.monotonic()
                chunk = policy.predict(qpos, prepared, hist)
                latency = time.monotonic() - since
                chunk_tick = tick
                queries.append((tick, chunk.copy(), latency))
                if args.temporal_agg:
                    ensemble = [(s, c) for s, c in ensemble if tick - s < len(c)] + [(tick, chunk)]
                    horizon = np.stack([ensemble_action(ensemble, tick + k, args.agg_k) for k in range(policy.chunk)
                                        if any(s <= tick + k < s + len(c) for s, c in ensemble)])
                    servo.publish(now, horizon[:, :7])
                else:
                    servo.publish(now, chunk[:, :7])
            action = ensemble_action(ensemble, tick, args.agg_k) if args.temporal_agg else chunk[min(tick - chunk_tick, len(chunk) - 1)]
            grip.update(float(action[7]), now - t0)
            hand = envelope.hand(state["q"])
            records.append(dict(t=now - t0, q=state["q"], dq=state["dq"], width=width, qpos=qpos, action=action,
                                hand=hand, q_ref=servo.q_ref.copy(),
                                images=policy.prepare_images(images) if args.save_images else np.zeros((0,), np.uint8),
                                skin=hist[-1] if hist is not None else np.zeros((0,), np.float32)))
            lifted = hand[2] - envelope.grasp_z
            holding = grip.closed and grip.pending is None and np.isfinite(width) and width > args.min_width
            if holding and lifted >= args.lift_success:
                held_since = held_since if held_since is not None else now
                if now - held_since >= args.hold_success:
                    reason = "held"
                    break
            else:
                held_since = None
            if now - t0 >= args.duration:
                break
            tick += 1
            time.sleep(max(0.0, t0 + tick / POLICY_HZ - time.monotonic()))
        servo.hold()                             # stop following the policy while the gripper settles
        grip.wait()
        time.sleep(0.2)
        width = float(gripper_reader.width)
        final_hand = envelope.hand(np.asarray(robot.state.q))
        servo.holding = grip.closed and np.isfinite(width) and width > args.min_width
        success = bool(servo.holding and final_hand[2] - envelope.grasp_z >= args.lift_success)
    except BaseException as exc:
        failure, success = exc, False
        servo.put_back_allowed = False           # never run scripted motion after an abort
        servo.hold()
        servo.policy_done.set()
        servo.join(5)
        if servo.is_alive():
            robot.stop()
            servo.join(3)
    else:
        servo.policy_done.set()
        servo.join(60)
    result = dict(success=success, end=reason if failure is None else "aborted",
                  error="" if failure is None else repr(failure), servo_error="" if servo.error is None else repr(servo.error),
                  holding=bool(servo.holding), final_width_m=float(gripper_reader.width),
                  max_lift_m=float(max((r["hand"][2] for r in records), default=np.nan) - envelope.grasp_z),
                  min_hand_z_above_grasp_m=float(min((r["hand"][2] for r in records), default=np.nan) - envelope.grasp_z),
                  max_tracking_error_rad=servo.max_error_seen, duration_s=records[-1]["t"] if records else 0.0,
                  gripper_events=grip.events, queries=len(queries),
                  mean_latency_s=float(np.mean([q[2] for q in queries])) if queries else None,
                  policy=policy.manifest["kind"], exec_horizon=args.exec_horizon, temporal_agg=args.temporal_agg)
    save(out, records, queries, servo.commands, result, t0)
    print(json.dumps({k: v for k, v in result.items() if k != "gripper_events"}), flush=True)
    if servo.error is not None and failure is None:
        raise RuntimeError(f"put-back failed: {servo.error}")
    return result


def ensemble_action(ensemble, tick, k):
    """ACT temporal ensembling: exp(-k * age) weights, the oldest prediction weighted most."""
    valid = [(s, c[tick - s]) for s, c in ensemble if s <= tick < s + len(c)]
    weights = np.exp(-k * np.arange(len(valid)))
    return np.average(np.stack([a for _, a in valid]), axis=0, weights=weights)


def save(out, records, queries, commands, result, t0):
    out.mkdir(parents=True, exist_ok=False)
    with h5py.File(out / "rollout.h5", "w") as f:
        for key in (records[0] if records else {}):
            value = np.stack([np.asarray(r[key]) for r in records])
            f.create_dataset(key, data=value, compression="lzf" if value.ndim > 2 else None)
        if queries:
            f.create_dataset("queries/tick", data=[q[0] for q in queries])
            f.create_dataset("queries/actions", data=np.stack([q[1] for q in queries]))
            f.create_dataset("queries/latency_s", data=[q[2] for q in queries])
        if commands:
            f.create_dataset("control/t", data=[c[0] - t0 for c in commands])
            f.create_dataset("control/q_ref", data=np.stack([c[1] for c in commands]))
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")


class ReplayCamera:
    """--sim: plays a recorded row's camera video by wall time, so the policy sees real images."""

    def __init__(self, name, row, start):
        import cv2

        self.name, self.start = name, start
        cap, self.frames = cv2.VideoCapture(str(Path(row) / f"episode_00000000_{name}.mp4")), []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            self.frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not self.frames:
            raise ValueError(f"{row}: no {name} video")

    def latest(self):
        i = min(int((time.monotonic() - self.start[0]) * POLICY_HZ), len(self.frames) - 1)
        return (self.frames[max(i, 0)], None, i, 0.0, time.monotonic()), i

    def close(self):
        pass


def render_trial(saved, block, log, path, fps=10):
    """sim_state log -> mp4 (side | top view) next to the trial's rollout.h5."""
    import cv2

    from .sim_franky import render_rollout

    frames = render_rollout(saved, pick.HOME, block, log)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"sim video: {path}", flush=True)


def add_args(p):
    p.add_argument("--model", type=Path, help="exported bundle (python model.py export pick)")
    p.add_argument("--check-model", action="store_true", help="load the bundle and run one dummy prediction")
    p.add_argument("--offline", nargs="+", type=Path, metavar="ROW_OR_BATCH",
                   help="open-loop check on recorded rows (episode dirs or batch dirs); no robot")
    p.add_argument("--sim", action="store_true", help="closed loop on the MuJoCo arm + block, cameras replayed")
    p.add_argument("--sim-frames", type=Path, help="row whose videos the sim cameras replay (default: first clean row)")
    p.add_argument("--sim-view", action="store_true")
    p.add_argument("--sim-video", action="store_true", help="render each sim trial to sim.mp4 (side | top view; needs EGL)")
    p.add_argument("--sim-block", type=float, default=0.024, help="sim cube size (m); the real block grips at 24 mm")
    p.add_argument("--home", action="store_true", help="only move the arm HOME (position control) and exit")
    p.add_argument("--host", default="172.16.0.2")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--exec-horizon", type=int, help="ticks per chunk before querying again (default: chunk, the sim protocol)")
    p.add_argument("--temporal-agg", action="store_true", help="ACT temporal ensembling (query every tick)")
    p.add_argument("--agg-k", type=float, default=0.01)
    p.add_argument("--duration", type=float, default=20.0, help="s of policy control per trial (recorded episodes: 15-17 s)")
    p.add_argument("--lift-success", type=float, default=0.08, help="m above the grasp height that counts as lifted")
    p.add_argument("--hold-success", type=float, default=1.0, help="s held lifted before the trial ends early")
    p.add_argument("--max-velocity", type=float, default=0.5, help="rad/s rate limit of the streamed reference")
    p.add_argument("--max-target-delta", type=float, default=0.15, help="abort if the policy target is this far from the arm (rad)")
    p.add_argument("--max-error", type=float, default=0.10, help="abort if a joint lags its reference more (rad)")
    p.add_argument("--max-align", type=float, default=0.25, help="largest joint offset to the lifted pose the put-back accepts (rad)")
    p.add_argument("--floor-margin", type=float, default=0.015, help="m the hand may go below the saved grasp height")
    p.add_argument("--box-margin", type=float, default=0.10, help="m around the HOME-to-grasp box the hand may use")
    p.add_argument("--watchdog", type=float, default=1.0, help="s past the end of a chunk before aborting")
    p.add_argument("--no-put-back", action="store_true", help="stop after the policy; leave the arm where it is")
    p.add_argument("--grasp-force", type=float, default=20.0)
    p.add_argument("--gripper-speed", type=float, default=0.05)
    p.add_argument("--min-width", type=float, default=0.003)
    p.add_argument("--approach", type=float, default=0.10)
    p.add_argument("--lift", type=float, default=0.10)
    p.add_argument("--place-gap", type=float, default=0.005)
    p.add_argument("--speed", type=float, default=0.05)
    p.add_argument("--dynamics", type=float, default=0.1)
    p.add_argument("--save-images", action="store_true", help="store the policy's RGB input in rollout.h5 (large)")
    p.add_argument("--device")
    p.add_argument("--control-cpu", type=int)
    p.add_argument("--allow-missing", action="store_true", help="PC-ACT: start even if some skin sensors are silent")
    p.add_argument("--warmup-timeout", type=float, default=5.0)
    p.add_argument("--output", type=Path, help="evaluation directory (default evaluations/pick_<model>_<utc>)")
    p.add_argument("--wrist-serial")
    p.add_argument("--exo-serial")
    p.add_argument("--camera", nargs=2, action="append", default=[], metavar=("NAME", "SERIAL"))
    p.add_argument("--no-camera-reset", action="store_true")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)


def rows_of(paths):
    rows = []
    for path in paths:
        path = Path(path)
        found = [path] if (path / "trajectory.h5").exists() else sorted(path.glob("episode_*"))
        for row in found:
            result = json.loads((row / "result.json").read_text())
            if result.get("clean_success"):
                rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.py eval pick", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    from . import live_dashboard

    live_dashboard.add_args(p)
    args = p.parse_args(argv)
    if args.home:
        from franky import JointMotion, Robot

        robot = Robot(args.host)
        robot.relative_dynamics_factor = args.dynamics
        pick.go_home(robot, JointMotion)
        print("arm at HOME")
        return 0
    if args.model is None:
        p.error("--model is required (export one with python model.py export pick)")
    if args.trials < 1 or args.duration <= 0:
        p.error("--trials and --duration must be positive")
    policy = PickPolicy(args.model, args.device)
    args.exec_horizon = 1 if args.temporal_agg else (args.exec_horizon or policy.chunk)
    if not 1 <= args.exec_horizon <= policy.chunk:
        p.error(f"--exec-horizon must be 1..{policy.chunk}")
    print(f"{policy.kind}: chunk {policy.chunk}, exec horizon {args.exec_horizon}, cameras {policy.cameras}, "
          f"{policy.width}x{policy.height}, device {policy.device}", flush=True)
    if args.check_model:
        hist = np.full((8, 40, 8, 8), NO_RETURN_M, np.float32) if policy.encoder is not None else None
        chunk = policy.predict(qpos_vector(pick.HOME, 0.08),
                               np.zeros((len(policy.cameras), policy.height, policy.width, 3), np.uint8), hist)
        print(f"check ok: chunk {chunk.shape}, first arm target {np.round(chunk[0, :7], 3).tolist()}, "
              f"gripper {chunk[0, 7]:.2f}")
        return 0
    if args.offline:
        rows = rows_of(args.offline)
        if not rows:
            p.error("no clean rows found")
        results = offline(policy, rows, args.exec_horizon)
        summary = {k: float(np.mean([r[k] for r in results])) for k in
                   ("chunk_arm_mae_rad", "first_step_max_err_rad", "executed_arm_max_err_rad", "gripper_agreement")}
        late = [r["close_tick_predicted"] - r["close_tick_recorded"] for r in results
                if r["close_tick_predicted"] is not None and r["close_tick_recorded"] is not None]
        summary.update(rows=len(results), close_tick_offset_mean=float(np.mean(late)) if late else None,
                       rows_never_closing=sum(r["close_tick_predicted"] is None for r in results))
        print("summary " + json.dumps(summary))
        return 0

    saved = json.loads(pick.POSE_FILE.read_text())
    plan_args = pick.parse_args(["--plan-only", "--start-noise", "0", "--approach-noise", "0", "--noise-std", "0",
                                 "--approach", str(args.approach), "--lift", str(args.lift),
                                 "--place-gap", str(args.place_gap), "--speed", str(args.speed)])
    plan = pick.Plan(saved, plan_args, np.random.default_rng(0))
    envelope = Envelope(saved, args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "evaluations" / f"pick_{Path(args.model).name}{'_sim' if args.sim else ''}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(dict(vars(args), manifest=policy.manifest), default=str, indent=2) + "\n")

    world = None
    start_clock = [time.monotonic()]
    if args.sim:
        from . import sim_franky

        world = sim_franky.install(saved, pick.HOME, args.sim_block, args.sim_view)
    from franky import JointMotion, Robot

    affinity = os.sched_getaffinity(0)
    results = []
    with ExitStack() as cleanup:
        if world is not None:
            cleanup.callback(world.join, 2.0)
            cleanup.callback(world.stop_event.set)
        cleanup.callback(os.sched_setaffinity, 0, affinity)
        control_cpus = None if args.sim else base.reserve_control_cpus(args.control_cpu)
        listener = None
        if policy.encoder is not None and not args.sim:
            sockets = base.open_sensor_sockets()
            for sock in sockets:
                cleanup.callback(sock.close)
            listener = base.SensorListener(sockets)
            listener.start()
            cleanup.callback(listener.join, 1.0)
            cleanup.callback(listener.stop_event.set)
            base.wait_for_sensors(listener, args.warmup_timeout, args.allow_missing)
        if args.sim:
            row = args.sim_frames or rows_of([ROOT / "recordings"] + sorted((ROOT / "recordings").glob("pick_*")))[0]
            cameras = [ReplayCamera(name, row, start_clock) for name in policy.cameras]
            print(f"sim cameras replay {row}", flush=True)
        else:
            cam_args = argparse.Namespace(**vars(args))
            cameras = base.open_cameras(cam_args)
            names = sorted(cam.name for cam in cameras)
            if names != sorted(policy.cameras):
                raise SystemExit(f"cameras {names} differ from the policy's {policy.cameras}")
        for cam in cameras:
            cleanup.callback(cam.close)
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = args.dynamics
        gripper = base.GripperReader(args.host)
        gripper.start()
        cleanup.callback(gripper.stop_event.set)
        cleanup.callback(live_dashboard.stop, live_dashboard.start(args, robot, gripper, listener, cameras, None,
                                                                    "pick eval" + (" (sim)" if args.sim else "")))
        for trial in range(args.trials):
            print(f"trial {trial + 1}/{args.trials}: home, gripper open (not part of the policy)", flush=True)
            gripper.gripper.open(args.gripper_speed)
            pick.go_home(robot, JointMotion)
            block_start = None
            if world is not None:
                block_start = world.block_position()
                world.reset_block_stats()
            start_clock[0] = time.monotonic()
            if world is not None:
                world.record_start()
            result = run_trial(robot, gripper, cameras, listener, policy, plan, envelope, args,
                               output / f"trial_{trial:04d}", control_cpus)
            if world is not None:
                log = world.record_stop()
                np.savez_compressed(output / f"trial_{trial:04d}" / "sim_state.npz",
                                    t=np.array([t for t, _ in log]), qpos=np.stack([q for _, q in log]) if log else np.zeros((0,)))
                if args.sim_video:
                    render_trial(saved, args.sim_block, log, output / f"trial_{trial:04d}" / "sim.mp4")
            if world is not None and block_start is not None:
                result["sim_block_rise_m"] = float(world.block_top - block_start[2])
                result["sim_block_put_back_offset_m"] = float(np.linalg.norm(world.block_position() - block_start))
                print(f"sim: block rose {result['sim_block_rise_m'] * 1000:.0f} mm, put back "
                      f"{result['sim_block_put_back_offset_m'] * 1000:.1f} mm from its start", flush=True)
                trial_dir = output / f"trial_{trial:04d}"
                (trial_dir / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
            results.append(result)
            if result["end"] == "aborted":
                print("aborted: the arm stopped where it was. Check it, then home it with "
                      "python model.py eval pick --model <bundle> --home", flush=True)
                break
        if results and results[-1]["end"] != "aborted":
            pick.go_home(robot, JointMotion)
    summary = dict(trials=len(results), successes=sum(r["success"] for r in results),
                   aborted=sum(r["end"] == "aborted" for r in results), output=str(output))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("summary " + json.dumps(summary), flush=True)
    return 0
