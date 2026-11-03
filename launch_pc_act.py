"""Launch a detached local PC-ACT training process and print its monitoring paths."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--epochs',type=int,default=2000)
    p.add_argument('--encoder-checkpoint',type=Path)
    p.add_argument('--initialization',choices=['scratch','act'],default='scratch')
    p.add_argument('--wandb-project',default='PC_ACT_experiments')
    p.add_argument('--wandb-entity')
    p.add_argument('--no-wandb',action='store_true')
    args=p.parse_args()
    run=(args.run_dir or root/'training_runs'/datetime.now(timezone.utc).strftime('pc_act_readout_%Y%m%dT%H%M%SZ')).resolve()
    if run.exists():raise FileExistsError(run)
    run.parent.mkdir(parents=True,exist_ok=True)
    # The trainer creates the unique run directory; keep its startup log beside it.
    log=run.parent/(run.name+'.log')
    command=[sys.executable,'-u',str(root/'train_pc_act.py'),'--run-dir',str(run),'--epochs',str(args.epochs),
             '--initialization',args.initialization]
    command+=['--wandb-project',args.wandb_project]
    if args.wandb_entity:command+=['--wandb-entity',args.wandb_entity]
    if args.no_wandb:command+=['--no-wandb']
    if args.encoder_checkpoint:command+=['--encoder-checkpoint',str(args.encoder_checkpoint)]
    with log.open('x') as output:
        process=subprocess.Popen(command,cwd=root,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=process.pid,run_dir=str(run),log=str(log),command=command,
              policy_initialization=args.initialization,
              encoder_initialization='pretrained' if args.encoder_checkpoint else 'random')
    (run.parent/(run.name+'.launch.json')).write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
