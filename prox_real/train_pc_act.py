"""Standalone PC-ACT readout overfit training on the recorded A -> B episodes.

Initializes the full policy and geometry encoder randomly by default. ACT warm-start
and pretrained geometry are optional, explicit alternatives recorded in config.
No collision-avoidance labels are synthesized. No robot connection is made.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from act_model import build_model
from act_model.sensor_order import HYBRID_SKIN_SENSOR_ORDER
from act_model.surface_geometry import SurfaceGeometryEncoder, save_encoder_checkpoint
from .eval_act_real import read_training_config

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def history_at(proximity,timestep):
    """Eight causal 15 Hz snapshots, each repeated four times; never use future data."""
    indices=np.maximum(np.arange(timestep-7,timestep+1),0)
    return np.repeat(proximity[indices].transpose(1,0,2,3),4,axis=1).copy()


def load_episodes(data,cameras,height,width):
    episodes=[]
    for path in sorted(data.glob('episode_*/trajectory.h5')):
        result=json.loads((path.parent/'result.json').read_text())
        if not result.get('clean_success'):raise ValueError(f'failed recording: {path}')
        with h5py.File(path) as f:
            g=f['traj_0'];q=g['real/q'][:].astype(np.float32);actions=g['real/target_q'][:].astype(np.float32)
            skin=np.stack([g['obs/proximity'][name][:].mean(axis=1) for name in HYBRID_SKIN_SENSOR_ORDER],axis=1).astype(np.float32)
            if not all(np.isfinite(a).all() for a in (q,actions,skin)):raise ValueError(f'nonfinite data: {path}')
            q=np.pad(q,((0,0),(0,2)));actions=np.pad(actions,((0,0),(0,1)))
            hashes=dict(trajectory=digest(path))
        videos=[]
        for name in cameras:
            vp=path.parent/f'episode_00000000_{name}.mp4'
            cap=cv2.VideoCapture(str(vp));frames=[]
            while True:
                ok,bgr=cap.read()
                if not ok:break
                frames.append(cv2.cvtColor(cv2.resize(bgr,(width,height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
            cap.release()
            if len(frames)!=len(q):raise ValueError(f'RGB count mismatch: {vp}')
            videos.append(np.asarray(frames));hashes[name]=digest(vp)
        images=np.stack(videos,axis=1)
        episodes.append(dict(name=path.parent.name,q=q,actions=actions,skin=skin,rgb=images,hashes=hashes))
        print(f'Loaded {path.parent.name}: {len(q)} observations; finite RGB/state/proximity',flush=True)
    if len(episodes)!=20:raise ValueError(f'expected the audited 20 episodes, found {len(episodes)}')
    return episodes


class Samples(Dataset):
    def __init__(self,episodes,ids,stats,chunk,validation=False):
        self.episodes,self.ids,self.stats,self.chunk=episodes,ids,stats,chunk
        self.fixed=[(i,t) for i in ids for t in (0,38,76,114)] if validation else None

    def __len__(self):return len(self.fixed) if self.fixed is not None else len(self.ids)

    def __getitem__(self,index):
        if self.fixed is not None:i,t=self.fixed[index]
        else:
            i=self.ids[index];t=int(torch.randint(len(self.episodes[i]['q']),(1,)))
        e=self.episodes[i];start=max(0,t-1)  # Match the ACT real-data loader used for the baseline.
        block=e['actions'][start:start+self.chunk]
        action=np.zeros((self.chunk,8),np.float32);action[:len(block)]=block
        pad=np.arange(self.chunk)>=len(block)
        q=(e['q'][t]-self.stats['qpos_mean'])/self.stats['qpos_std']
        action=(action-self.stats['action_mean'])/self.stats['action_std']
        return (torch.from_numpy(e['rgb'][t].transpose(0,3,1,2).copy()),torch.from_numpy(q),
                torch.from_numpy(action),torch.from_numpy(pad),torch.from_numpy(history_at(e['skin'],t)))


def initialize_policy(config,checkpoint):
    model=build_model(config)
    old=torch.load(checkpoint,map_location='cpu',weights_only=True)
    old={k.removeprefix('model.'):v for k,v in old.items()}
    with torch.no_grad():
        model.additional_pos_embed.weight[:2].copy_(old.pop('additional_pos_embed.weight'))
        model.additional_pos_embed.weight[2:].zero_()
    missing,unexpected=model.load_state_dict(old,strict=False)
    allowed={'additional_pos_embed.weight','input_proj_proximity.weight','input_proj_proximity.bias'}
    if set(missing)!=allowed or unexpected:raise ValueError(f'warm-start mismatch: {missing}, {unexpected}')
    return model


def prepare_batch(batch,device):
    rgb,q,action,pad,skin=[value.to(device,non_blocking=True) for value in batch]
    rgb=rgb.float()/255
    mean=rgb.new_tensor([.485,.456,.406]).view(1,1,3,1,1)
    std=rgb.new_tensor([.229,.224,.225]).view(1,1,3,1,1)
    return (rgb-mean)/std,q,action,pad,skin


def losses(pred,action,pad,mu=None,logvar=None,kl_weight=10):
    mask=(~pad).unsqueeze(-1)
    # Original ACT objective averages masked L1 over the complete padded tensor.
    l1=(abs(pred-action)*mask).mean()
    kl=pred.new_zeros(()) if mu is None else (-.5*(1+logvar-mu.square()-logvar.exp()).sum(1)).mean()
    return l1+kl_weight*kl,l1,kl


@torch.inference_mode()
def evaluate(model,encoder,loader,device,action_std,ablations=False):
    model.eval();encoder.eval()
    errors={k:0. for k in (['live','clear','shuffled','near_delta'] if ablations else ['live'])}
    counts={k:0 for k in errors};l1s=[]
    for batch in loader:
        rgb,q,action,pad,skin=prepare_batch(batch,device)
        features=encoder(skin)
        pred,_,_=model(q,rgb,None,proximity_positions=features)
        mask=(~pad).unsqueeze(-1)
        l1s.append(float((abs(pred-action)*mask).sum()/mask.sum()/action.shape[-1]))
        arm_mask=mask.expand(-1,-1,7)
        errors['live']+=float((abs(pred[:,:,:7]-action[:,:,:7])*action_std[:7]*arm_mask).sum())
        counts['live']+=int(arm_mask.sum())
        if ablations:
            for name,changed in [('clear',torch.full_like(skin,4.)),('shuffled',skin.roll(1,0))]:
                other,_,_=model(q,rgb,None,proximity_positions=encoder(changed))
                errors[name]+=float((abs(other[:,:,:7]-action[:,:,:7])*action_std[:7]*arm_mask).sum())
                counts[name]+=int(arm_mask.sum())
            near=skin.clone();near[:,:4]=.05  # diagnostic only: synthetic obstruction at first four sensors
            other,_,_=model(q,rgb,None,proximity_positions=encoder(near))
            errors['near_delta']+=float((abs(other[:,:,:7]-pred[:,:,:7])*action_std[:7]*arm_mask).sum())
            counts['near_delta']+=int(arm_mask.sum())
    return dict(normalized_l1=float(np.mean(l1s)),**{k+'_rad':errors[k]/max(counts[k],1) for k in errors})


def save_pair(run,label,model,encoder,metrics,epoch):
    # Both files carry the same epoch/metrics; no encoder from another epoch may be paired.
    torch.save({'model.'+k:v.detach().cpu() for k,v in model.state_dict().items()},run/f'policy_{label}.ckpt')
    save_encoder_checkpoint(run/f'prox_encoder_{label}.pt',encoder.inner,kind='embedding',
                            frozen=False,extra=dict(policy_tap='readout',epoch=epoch,metrics=metrics))


def detach(argv):
    """Start this trainer as a detached process; log and launch info beside the run directory."""
    import subprocess
    import sys

    argv=[a for a in argv if a!='--detach']
    if '--run-dir' in argv:run=Path(argv[argv.index('--run-dir')+1])
    else:
        run=ROOT/'training_runs'/datetime.now(timezone.utc).strftime('pc_act_readout_%Y%m%dT%H%M%SZ')
        argv=['--run-dir',str(run),*argv]
    run=run.resolve()
    if run.exists():raise FileExistsError(run)
    run.parent.mkdir(parents=True,exist_ok=True)
    # The trainer creates the unique run directory; keep its startup log beside it.
    log=run.parent/(run.name+'.log')
    command=[sys.executable,'-u',str(ROOT/'model.py'),'train',*argv]
    with log.open('x') as output:
        process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=process.pid,run_dir=str(run),log=str(log),command=command)
    (run.parent/(run.name+'.launch.json')).write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


def main(argv=None):
    if argv is not None and '--detach' in argv:return detach(argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'recordings/act_test_20')
    p.add_argument('--act-checkpoint',type=Path,default=ROOT/'models/act_test_20')
    p.add_argument('--encoder-checkpoint',type=Path)
    p.add_argument('--initialization',choices=['scratch','act'],default='scratch')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--detach',action='store_true',help='run in the background (run dir defaults to training_runs/pc_act_readout_<utc>)')
    p.add_argument('--epochs',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--encoder-lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--split-seed',type=int,default=1)
    p.add_argument('--validate-every',type=int,default=25)
    p.add_argument('--ablate-every',type=int,default=100)
    p.add_argument('--device',default='cuda')
    p.add_argument('--wandb-project',default='PC_ACT_experiments')
    p.add_argument('--wandb-entity')
    p.add_argument('--no-wandb',action='store_true',help='write local metrics only')
    args=p.parse_args(argv)
    if args.epochs<1 or args.batch_size<1 or args.validate_every<1 or args.ablate_every<1:p.error('counts must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; training must be launched with GPU access')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.run_dir.mkdir(parents=True,exist_ok=False)
    run=args.run_dir;started=time.time()
    cfg=read_training_config(args.act_checkpoint)
    cfg.update(n_proximity_sensors=40,prox_tokens_per_sensor=1,prox_feat_dim=128)
    cameras=cfg['camera_names'];meta=json.loads((args.act_checkpoint/'image_metadata.json').read_text())
    with np.load(args.act_checkpoint/'dataset_stats.npz',allow_pickle=False) as f:stats={k:f[k].copy() for k in f.files}
    episodes=load_episodes(args.data,cameras,meta['image_h'],meta['image_w'])
    order=np.random.RandomState(args.split_seed).permutation(len(episodes)).tolist()
    train_ids,val_ids=order[:16],order[16:]
    provenance=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        policy_config=cfg,train_episode_ids=train_ids,validation_episode_ids=val_ids,
        policy_initialization=args.initialization,
        warm_start_act_sha256=digest(args.act_checkpoint/'policy_best.ckpt') if args.initialization=='act' else None,
        encoder_initialization='pretrained' if args.encoder_checkpoint else 'random',
        encoder_initial_sha256=digest(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        normalization_provenance='Reused original ACT stats (all 20 original episodes); validation is descriptive, not an unbiased generalization estimate.',
        input_semantics='causal 8x15Hz snapshots, each repeated four times; 20cm closeness cap; 40x128 readout tokens',
        data_hashes={e['name']:e['hashes'] for e in episodes},
        scientific_limit='Unobstructed trajectory imitation only. No collision avoidance demonstrated by low action loss or synthetic sensor sensitivity.')
    (run/'training_config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (run/'image_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    np.savez(run/'dataset_stats.npz',**stats)
    (run/'prox_config.json').write_text(json.dumps(dict(use_proximity=True,prox_feature='surface_embedding',
        prox_policy_tap='readout',prox_feat_dim=128,n_proximity_sensors=40,prox_tokens_per_sensor=1,
        finetune_prox_encoder=True,prox_encoder_finetuned='prox_encoder_best.pt',sensor_order=list(HYBRID_SKIN_SENSOR_ORDER),
        history_sampling='consecutive_control_steps_v1',proximity_layout='raw_causal'),indent=2)+'\n')
    model=(initialize_policy(cfg,args.act_checkpoint/'policy_best.ckpt') if args.initialization=='act'
           else build_model(cfg)).to(args.device)
    encoder=SurfaceGeometryEncoder(kind='embedding',checkpoint=args.encoder_checkpoint,device=args.device,
                                   frozen=False,policy_tap='readout')
    params=[dict(params=[v for k,v in model.named_parameters() if 'backbone' not in k],lr=args.lr),
            dict(params=[v for k,v in model.named_parameters() if 'backbone' in k],lr=1e-5),
            dict(params=encoder.parameters(),lr=args.encoder_lr)]
    optimizer=torch.optim.AdamW(params,weight_decay=1e-4)
    training=DataLoader(Samples(episodes,train_ids,stats,cfg['num_queries']),batch_size=args.batch_size,shuffle=True,num_workers=0)
    validation=DataLoader(Samples(episodes,val_ids,stats,cfg['num_queries'],True),batch_size=4,shuffle=False,num_workers=0)
    std=torch.as_tensor(stats['action_std'],device=args.device)
    best=float('inf');stop=[False]
    def interrupt(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    wandb_run=None
    if not args.no_wandb:
        from .pc_act_wandb import start_run
        wandb_run=start_run(run,provenance,args.wandb_project,args.wandb_entity)
    print(f'PC-ACT run: {run}; device={torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device}; '
          f'train={train_ids}, val={val_ids}; policy_init={args.initialization}; '
          f'encoder_init={provenance["encoder_initialization"]}',flush=True)
    epoch=-1;latest={}
    try:
        for epoch in range(args.epochs):
            if stop[0]:break
            model.train();encoder.train();train_losses=[]
            for batch in training:
                rgb,q,action,pad,skin=prepare_batch(batch,args.device)
                optimizer.zero_grad(set_to_none=True)
                pred,_,(mu,logvar)=model(q,rgb,None,action,pad,proximity_positions=encoder(skin))
                loss,l1,kl=losses(pred,action,pad,mu,logvar)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(encoder.parameters()),1.)
                optimizer.step();train_losses.append([float(loss),float(l1),float(kl)])
            latest=dict(epoch=epoch,train_loss=float(np.mean([x[0] for x in train_losses])),
                        train_l1=float(np.mean([x[1] for x in train_losses])),train_kl=float(np.mean([x[2] for x in train_losses])),
                        elapsed_s=time.time()-started)
            if epoch%args.validate_every==0 or epoch==args.epochs-1:
                latest['validation']=evaluate(model,encoder,validation,args.device,std,
                                              ablations=epoch%args.ablate_every==0 or epoch==args.epochs-1)
                metric=latest['validation']['normalized_l1']
                if metric<best:
                    best=metric;save_pair(run,'best',model,encoder,latest,epoch)
                save_pair(run,'last',model,encoder,latest,epoch)
                (run/'progress.json').write_text(json.dumps(dict(latest,best_validation_l1=best,pid=os.getpid(),status='running'),indent=2)+'\n')
                print(json.dumps(latest),flush=True)
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(latest)+'\n')
            if wandb_run is not None:
                from .pc_act_wandb import flatten_metrics
                wandb_run.log(flatten_metrics(latest),step=epoch)
    finally:
        save_pair(run,'last',model,encoder,latest,epoch)
        (run/'progress.json').write_text(json.dumps(dict(latest,best_validation_l1=best,pid=os.getpid(),
            status='interrupted' if stop[0] else ('complete' if epoch==args.epochs-1 else 'failed')),indent=2)+'\n')
        if wandb_run is not None:
            from .pc_act_wandb import read_history,set_summary
            progress=json.loads((run/'progress.json').read_text())
            if (run/'metrics.jsonl').exists():set_summary(wandb_run,run,read_history(run),progress)
            wandb_run.finish(exit_code=0 if progress['status']=='complete' else 1)
