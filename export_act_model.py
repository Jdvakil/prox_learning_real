"""Copy an ACT checkpoint and portable deployment metadata into this repository."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from eval_act_real import read_training_config


def export(source, destination, data_dir=None, image_height=None, image_width=None):
    config = read_training_config(source)
    if not config.get("camera_names") or not config.get("nheads"):
        raise ValueError("source must contain training_config.json or W&B training metadata")
    meta = {}
    for path in [source / "image_metadata.json", data_dir / "convert_meta.json" if data_dir else source / "convert_meta.json"]:
        if path.exists():
            meta = json.loads(path.read_text())
            break
    height = image_height or meta.get("image_h")
    width = image_width or meta.get("image_w")
    if not height or not width:
        raise ValueError("provide --data-dir or explicit --image-height and --image-width")
    if meta.get("camera_names") and config["camera_names"] != meta["camera_names"]:
        raise ValueError("training and image metadata camera orders differ")
    stats_npz = source / "dataset_stats.npz"
    if stats_npz.exists():
        with np.load(stats_npz, allow_pickle=False) as f:
            stats = {key:f[key].copy() for key in ("qpos_mean","qpos_std","action_mean","action_std")}
    else:
        with (source / "dataset_stats.pkl").open("rb") as f:
            legacy = pickle.load(f)
        stats = {key:np.asarray(legacy[key],dtype=np.float32) for key in
                 ("qpos_mean","qpos_std","action_mean","action_std")}
    for key,value in stats.items():
        if not np.isfinite(value).all():raise ValueError(f"invalid stats {key}")
    checkpoint = source / "policy_best.ckpt"
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copy2(checkpoint,destination/checkpoint.name)
    np.savez(destination/"dataset_stats.npz",**stats)
    (destination/"training_config.json").write_text(json.dumps(dict(policy_config=config),indent=2)+"\n")
    (destination/"image_metadata.json").write_text(json.dumps(dict(
        camera_names=config["camera_names"],image_h=int(height),image_w=int(width)),indent=2)+"\n")
    manifest={}
    for path in sorted(destination.iterdir()):
        hasher=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):hasher.update(block)
        manifest[path.name]=dict(bytes=path.stat().st_size,sha256=hasher.hexdigest())
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Exported self-contained model: {destination}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--destination",type=Path,default=Path(__file__).resolve().parent/"models/act_test_20")
    p.add_argument("--data-dir",type=Path)
    p.add_argument("--image-height",type=int)
    p.add_argument("--image-width",type=int)
    args=p.parse_args()
    export(args.source,args.destination,args.data_dir,args.image_height,args.image_width)
