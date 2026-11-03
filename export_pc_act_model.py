"""Export a paired PC-ACT checkpoint and recorded background for standalone deployment."""
import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch

from act_model.sensor_order import HYBRID_SKIN_SENSOR_ORDER
from export_act_model import export
from eval_pc_act_real import sha256

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'training_runs/pc_act_test_20_scratch_20261103')
    p.add_argument('--destination',type=Path,default=ROOT/'models/pc_act_test_20')
    p.add_argument('--background-data',type=Path,default=ROOT/'recordings/act_test_20')
    args=p.parse_args()
    encoder=args.source/'prox_encoder_best.pt'
    payload=torch.load(encoder,map_location='cpu',weights_only=True)
    if payload.get('policy_tap')!='readout' or payload.get('policy_feature_dim')!=128:raise ValueError('expected finetuned PC-ACT readout encoder')
    export(args.source,args.destination)
    for filename in ['prox_encoder_best.pt','prox_config.json','training_config.json']:
        shutil.copy2(args.source/filename,args.destination/filename)
    names=list(HYBRID_SKIN_SENSOR_ORDER);background=np.full((40,8,8),4.,np.float32)
    rows=sorted(args.background_data.glob('episode_*/trajectory.h5'))
    if not rows:raise ValueError('recorded background episodes missing')
    for path in rows:
        with h5py.File(path) as f:
            values=np.stack([f['traj_0/obs/proximity/'+name][:].mean(1) for name in names],axis=1)
            values=np.where((values>=.005)&(values<4),values,4.)
            background=np.minimum(background,values.min(0))
    np.savez(args.destination/'guard_baseline.npz',minimum_depth_m=background,sensor_names=np.asarray(names))
    pair=dict(policy_file='policy_best.ckpt',encoder_file='prox_encoder_best.pt',epoch=payload['epoch'],
              policy_sha256=sha256(args.destination/'policy_best.ckpt'),encoder_sha256=sha256(args.destination/'prox_encoder_best.pt'))
    (args.destination/'policy_pair.json').write_text(json.dumps(pair,indent=2)+'\n')
    manifest={path.name:dict(bytes=path.stat().st_size,sha256=sha256(path))
              for path in sorted(args.destination.iterdir()) if path.name!='manifest.json'}
    (args.destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'PC-ACT epoch {pair["epoch"]} and background from {len(rows)} episodes exported to {args.destination}')


if __name__=='__main__':main()
