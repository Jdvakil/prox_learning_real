"""Copy a trained ACT or PC-ACT checkpoint into a self-contained deployment bundle under models/.

    python model.py export act --source <run dir>              # -> models/act_test_20
    python model.py export pc-act --source <run dir>           # -> models/pc_act_test_20
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import h5py
import numpy as np

from .eval_act_real import read_training_config

ROOT = Path(__file__).resolve().parents[1]


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
        manifest[path.name]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Exported self-contained model: {destination}")


def sha256(path):
    hasher=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):hasher.update(block)
    return hasher.hexdigest()


def export_pc_act(source, destination, background_data):
    """ACT export plus the paired geometry encoder and the recorded proximity background."""
    import torch

    from act_model.sensor_order import HYBRID_SKIN_SENSOR_ORDER

    payload=torch.load(source/'prox_encoder_best.pt',map_location='cpu',weights_only=True)
    if payload.get('policy_tap')!='readout' or payload.get('policy_feature_dim')!=128:raise ValueError('expected finetuned PC-ACT readout encoder')
    export(source,destination)
    for filename in ['prox_encoder_best.pt','prox_config.json','training_config.json']:
        shutil.copy2(source/filename,destination/filename)
    names=list(HYBRID_SKIN_SENSOR_ORDER);background=np.full((40,8,8),4.,np.float32)
    rows=sorted(background_data.glob('episode_*/trajectory.h5'))
    if not rows:raise ValueError('recorded background episodes missing')
    for path in rows:
        with h5py.File(path) as f:
            values=np.stack([f['traj_0/obs/proximity/'+name][:].mean(1) for name in names],axis=1)
            values=np.where((values>=.005)&(values<4),values,4.)
            background=np.minimum(background,values.min(0))
    np.savez(destination/'guard_baseline.npz',minimum_depth_m=background,sensor_names=np.asarray(names))
    pair=dict(policy_file='policy_best.ckpt',encoder_file='prox_encoder_best.pt',epoch=payload['epoch'],
              policy_sha256=sha256(destination/'policy_best.ckpt'),encoder_sha256=sha256(destination/'prox_encoder_best.pt'))
    (destination/'policy_pair.json').write_text(json.dumps(pair,indent=2)+'\n')
    manifest={path.name:dict(bytes=path.stat().st_size,sha256=sha256(path))
              for path in sorted(destination.iterdir()) if path.name!='manifest.json'}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'PC-ACT epoch {pair["epoch"]} and background from {len(rows)} episodes exported to {destination}')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind",choices=["act","pc-act","pick"])
    p.add_argument("--checkpoint",default="policy_best.ckpt",help="pick: policy file in the run directory")
    p.add_argument("--source",type=Path,help="training run directory (pc-act default: training_runs/pc_act_test_20_scratch_20261103)")
    p.add_argument("--destination",type=Path,help="default models/act_test_20 or models/pc_act_test_20")
    p.add_argument("--data-dir",type=Path,help="act: converted dataset with convert_meta.json (image size)")
    p.add_argument("--image-height",type=int)
    p.add_argument("--image-width",type=int)
    p.add_argument("--background-data",type=Path,default=ROOT/"recordings/act_test_20",
                   help="pc-act: recorded episodes for the proximity guard background")
    args=p.parse_args(argv)
    if args.kind=="pick":
        from .eval_pick import export_bundle
        if args.source is None or args.destination is None:p.error("pick export needs --source and --destination")
        return export_bundle(args.source,args.destination,args.checkpoint,args.data_dir)
    if args.kind=="act":
        if args.source is None:p.error("act export needs --source")
        export(args.source,args.destination or ROOT/"models/act_test_20",args.data_dir,args.image_height,args.image_width)
    else:
        export_pc_act(args.source or ROOT/"training_runs/pc_act_test_20_scratch_20261103",
                      args.destination or ROOT/"models/pc_act_test_20",args.background_data)
