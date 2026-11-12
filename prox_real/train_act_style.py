"""Train ACT or PC-ACT on ACT-style episode HDF5s (scripts/convert_pact_place_to_act.py output).

Both arms use the same bundled ACT, split, normalization, sampling and schedule; the only
difference is the proximity input. --arm act is cameras + joints. --arm pc-act adds the
40-sensor skin through the surface encoder (shared conv stem per 8x8 frame, then a temporal
transformer over eight causal 15 Hz snapshots, 40 x 128 readout tokens), trained end to end.

    python model.py train-act-style --arm act    --data <act dir> --rows <rows dir> --run-dir training_runs/pick_act
    python model.py train-act-style --arm pc-act --data <act dir> --rows <rows dir> --run-dir training_runs/pick_pcact

An epoch is one pass over every training frame (shuffled). The validation episodes are whole
noise draws: every episode that shares a start pose with a validation episode is held out too,
so validation starts were never seen in training. Normalization statistics come from the
training episodes only. metrics.jsonl gets one line per epoch; progress.json the latest state.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from act_model import build_model
from act_model.surface_geometry import SurfaceGeometryEncoder
from .train_pc_act import history_at, losses, prepare_batch, save_pair

CAMERAS = ["exo_camera_1", "wrist_camera"]


def episode_groups(rows, count):
    """Group id per converted episode: rows sorted by name, failed rows skipped (as the converter does)."""
    keys = []
    for row in sorted(Path(rows).iterdir()):
        result = json.loads((row / "result.json").read_text())
        if not result.get("clean_success"):
            continue
        with h5py.File(row / "trajectory.h5") as f:
            start = f["traj_0"].attrs.get("start_q")
        keys.append(row.name if start is None else tuple(np.round(start, 4)))
    if len(keys) != count:
        raise ValueError(f"{len(keys)} clean rows but {count} converted episodes")
    ids = {}
    return [ids.setdefault(key, len(ids)) for key in keys]


def load(data):
    paths = sorted(Path(data).glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
    episodes = []
    for path in paths:
        with h5py.File(path) as f:
            episodes.append(dict(
                name=path.stem, q=f["observations/qpos"][:].astype(np.float32),
                action=f["action"][:].astype(np.float32),
                skin=f["observations/proximity"][:].astype(np.float32),
                rgb=np.stack([f[f"observations/images/{c}"][:] for c in CAMERAS], axis=1)))
    print(f"loaded {len(episodes)} episodes, {sum(len(e['q']) for e in episodes)} frames", flush=True)
    return episodes


def statistics(episodes):
    q = np.concatenate([e["q"] for e in episodes])
    a = np.concatenate([e["action"] for e in episodes])
    return dict(qpos_mean=q.mean(0), qpos_std=np.clip(q.std(0), 1e-2, None),
                action_mean=a.mean(0), action_std=np.clip(a.std(0), 1e-2, None))


class Frames(Dataset):
    """Every (episode, t) once; the action chunk starts at t, padded past the episode end."""

    def __init__(self, episodes, ids, stats, chunk, pc):
        self.episodes, self.stats, self.chunk, self.pc = episodes, stats, chunk, pc
        self.index = [(i, t) for i in ids for t in range(len(episodes[i]["q"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        e = self.episodes[i]
        block = e["action"][t:t + self.chunk]
        action = np.zeros((self.chunk, block.shape[1]), np.float32)
        action[:len(block)] = block
        pad = np.arange(self.chunk) >= len(block)
        q = (e["q"][t] - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        action = (action - self.stats["action_mean"]) / self.stats["action_std"]
        skin = history_at(e["skin"], t) if self.pc else np.zeros(1, np.float32)
        return (torch.from_numpy(e["rgb"][t].transpose(0, 3, 1, 2).copy()), torch.from_numpy(q.astype(np.float32)),
                torch.from_numpy(action.astype(np.float32)), torch.from_numpy(pad), torch.from_numpy(skin))


@torch.inference_mode()
def validate(model, encoder, loader, device, std):
    model.eval()
    if encoder is not None:
        encoder.eval()
    l1 = arm = grip = n_arm = n_grip = 0.0
    batches = 0
    for batch in loader:
        rgb, q, action, pad, skin = prepare_batch(batch, device)
        pred, _, _ = model(q, rgb, None, proximity_positions=None if encoder is None else encoder(skin))
        mask = (~pad).unsqueeze(-1)
        l1 += float((abs(pred - action) * mask).sum() / mask.sum() / action.shape[-1])
        batches += 1
        err = abs(pred - action) * std
        arm += float((err[:, :, :7] * mask).sum())
        n_arm += float(mask.sum()) * 7
        grip += float((err[:, :, 7:] * mask).sum())
        n_grip += float(mask.sum())
    return dict(val_l1=l1 / batches, val_arm_rad=arm / n_arm, val_gripper=grip / n_grip)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["act", "pc-act"], required=True)
    p.add_argument("--data", type=Path, required=True, help="converted episode_*.hdf5 directory")
    p.add_argument("--rows", type=Path, required=True, help="the raw rows the converter read (for the split)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--val-groups", type=int, default=3, help="noise draws (start poses) held out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    run, started, pc = args.run_dir, time.time(), args.arm == "pc-act"

    episodes = load(args.data)
    groups = episode_groups(args.rows, len(episodes))
    held = set(np.random.RandomState(args.split_seed).permutation(max(groups) + 1)[:args.val_groups].tolist())
    val_ids = [i for i, g in enumerate(groups) if g in held]
    train_ids = [i for i, g in enumerate(groups) if g not in held]
    stats = statistics([episodes[i] for i in train_ids])
    cfg = dict(policy_class="ACT", backbone="resnet18", camera_names=CAMERAS, num_queries=args.chunk,
               hidden_dim=512, dim_feedforward=3200, enc_layers=4, dec_layers=7, nheads=8, kl_weight=10,
               lr=args.lr, lr_backbone=1e-5, state_dim=9, action_dim=8, seed=args.seed)
    if pc:
        cfg.update(n_proximity_sensors=40, prox_tokens_per_sensor=1, prox_feat_dim=128)
    (run / "training_config.json").write_text(json.dumps(dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, policy_config=cfg,
        train_episode_ids=train_ids, validation_episode_ids=val_ids, groups=groups,
        normalization="training episodes only", epoch="one pass over every training frame"), indent=2) + "\n")
    (run / "image_metadata.json").write_text(json.dumps(dict(camera_names=CAMERAS, image_h=240, image_w=320)) + "\n")
    np.savez(run / "dataset_stats.npz", **stats)

    model = build_model(cfg).to(args.device)
    encoder = SurfaceGeometryEncoder(kind="embedding", device=args.device, frozen=False,
                                     policy_tap="readout") if pc else None
    params = [dict(params=[v for k, v in model.named_parameters() if "backbone" not in k], lr=args.lr),
              dict(params=[v for k, v in model.named_parameters() if "backbone" in k], lr=1e-5)]
    if pc:
        params.append(dict(params=encoder.parameters(), lr=args.lr))
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    training = DataLoader(Frames(episodes, train_ids, stats, args.chunk, pc), batch_size=args.batch_size,
                          shuffle=True, drop_last=True, **common)
    # Every 4th validation frame keeps validation to a few seconds.
    val_set = Frames(episodes, val_ids, stats, args.chunk, pc)
    val_set.index = val_set.index[::4]
    validation = DataLoader(val_set, batch_size=32, shuffle=False, **common)
    std = torch.as_tensor(stats["action_std"], device=args.device)
    print(f"{args.arm}: train {len(train_ids)} episodes ({len(training.dataset)} frames), "
          f"val {len(val_ids)} episodes (held-out start poses), {len(training)} steps/epoch", flush=True)

    stop = [False]
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
    best, epoch, latest = float("inf"), -1, {}
    try:
        for epoch in range(args.epochs):
            if stop[0]:
                break
            model.train()
            if pc:
                encoder.train()
            sums = np.zeros(3)
            for step, batch in enumerate(training):
                rgb, q, action, pad, skin = prepare_batch(batch, args.device)
                pred, _, (mu, logvar) = model(q, rgb, None, action, pad,
                                              proximity_positions=encoder(skin) if pc else None)
                loss, l1, kl = losses(pred, action, pad, mu, logvar)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([v for g in params for v in g["params"]], 1.0)
                optimizer.step()
                sums += [float(loss), float(l1), float(kl)]
                if stop[0]:
                    break
            n = step + 1
            latest = dict(epoch=epoch + 1, train_loss=sums[0] / n, train_l1=sums[1] / n, train_kl=sums[2] / n,
                          elapsed_s=round(time.time() - started, 1),
                          **validate(model, encoder, validation, args.device, std))
            if latest["val_l1"] < best:
                best = latest["val_l1"]
                if pc:
                    save_pair(run, "best", model, encoder, latest, epoch)
                else:
                    torch.save({"model." + k: v.cpu() for k, v in model.state_dict().items()}, run / "policy_best.ckpt")
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(latest) + "\n")
            (run / "progress.json").write_text(json.dumps(dict(latest, epochs=args.epochs, best_val_l1=best,
                                                               pid=os.getpid(), status="running"), indent=2) + "\n")
            print(json.dumps(latest), flush=True)
    finally:
        if pc:
            save_pair(run, "last", model, encoder, latest, epoch)
        else:
            torch.save({"model." + k: v.cpu() for k, v in model.state_dict().items()}, run / "policy_last.ckpt")
        status = "interrupted" if stop[0] else ("complete" if epoch == args.epochs - 1 else "failed")
        (run / "progress.json").write_text(json.dumps(dict(latest, epochs=args.epochs, best_val_l1=best,
                                                           pid=os.getpid(), status=status), indent=2) + "\n")
