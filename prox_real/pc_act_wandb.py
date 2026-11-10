"""W&B logging and backfill of the standalone trainer's original local metrics."""
import argparse
import json
from pathlib import Path

DEFAULT_PROJECT = 'PC_ACT_experiments'


def flatten_metrics(row):
    values = {'epoch': row['epoch']}
    for source, target in [('train_loss','train/loss'),('train_l1','train/l1'),
                           ('train_kl','train/kl'),('elapsed_s','timing/elapsed_s')]:
        if source in row: values[target] = row[source]
    mapping = {'normalized_l1':'validation/normalized_l1','live_rad':'validation/arm_mae_rad',
               'clear_rad':'ablation/clear_arm_mae_rad','shuffled_rad':'ablation/shuffled_arm_mae_rad',
               'near_delta_rad':'ablation/synthetic_near_action_delta_rad'}
    for source, value in row.get('validation', {}).items():
        values[mapping.get(source, 'validation/'+source)] = value
    return values


def read_history(run_dir):
    rows = [json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines() if line.strip()]
    epochs = [row['epoch'] for row in rows]
    if not rows or any(b <= a for a,b in zip(epochs,epochs[1:])):
        raise ValueError('metrics must have strictly increasing epoch numbers')
    return rows


def start_run(run_dir, config, project=DEFAULT_PROJECT, entity=None, backfilled=False):
    import wandb

    run_dir = Path(run_dir).resolve()
    pointer = run_dir/'wandb_run.json'
    saved = json.loads(pointer.read_text()) if pointer.exists() else {}
    if saved and (saved['project'] != project or (entity and saved.get('entity') != entity)):
        raise ValueError('W&B destination differs from the saved run binding')
    run_id = saved.get('id') or wandb.util.generate_id()
    pointer.write_text(json.dumps(dict(saved,id=run_id,project=project,entity=entity or saved.get('entity')),indent=2)+'\n')
    settings = wandb.Settings(x_disable_stats=backfilled, init_timeout=60)
    run = wandb.init(project=project,entity=entity,id=run_id,resume='allow',name=run_dir.name,
        dir=str(run_dir),job_type='historical-metrics-backfill' if backfilled else 'pc-act-training',
        tags=['pc-act','trajectory-overfit',config.get('policy_initialization','unknown')]+(['backfilled'] if backfilled else []),
        config=dict(config,logging_backfilled=backfilled),settings=settings,save_code=False)
    run.define_metric('epoch')
    for prefix in ['train/*','validation/*','ablation/*','timing/*']:
        run.define_metric(prefix,step_metric='epoch')
    pointer.write_text(json.dumps(dict(saved,id=run.id,project=run.project,entity=run.entity,url=run.url),indent=2)+'\n')
    print(f'W&B run: {run.url}',flush=True)
    return run


def set_summary(run, run_dir, rows, progress):
    validations = [r for r in rows if 'validation' in r]
    best = min(validations,key=lambda r:r['validation']['normalized_l1']) if validations else None
    run.summary.update(dict(training_status=progress.get('status'),epochs_recorded=len(rows),
        last_epoch=rows[-1]['epoch'],checkpoint_directory=str(run_dir),
        scientific_limit='Trajectory imitation and sensor sensitivity; no collision avoidance established.'))
    if best:
        run.summary.update(dict(best_epoch=best['epoch'],best_validation_l1=best['validation']['normalized_l1'],
                                best_validation_arm_mae_rad=best['validation']['live_rad']))


def backfill(run_dir, project=DEFAULT_PROJECT, entity=None, dry_run=False):
    run_dir = Path(run_dir).resolve()
    rows = read_history(run_dir)
    config = json.loads((run_dir/'training_config.json').read_text())
    progress = json.loads((run_dir/'progress.json').read_text())
    if progress.get('status') == 'running':
        raise ValueError('backfill is for finished jobs; use live trainer logging for new runs')
    if dry_run:
        print(json.dumps(dict(project=project,entity=entity,rows=len(rows),first_epoch=rows[0]['epoch'],
                              last_epoch=rows[-1]['epoch'],training_status=progress['status'],
                              first_metrics=flatten_metrics(rows[0])),indent=2))
        return
    pointer = run_dir/'wandb_run.json'
    saved = json.loads(pointer.read_text()) if pointer.exists() else {}
    if saved.get('synced_through_epoch',-1) >= rows[-1]['epoch']:
        print('Already synced:',saved.get('url'));return
    run = start_run(run_dir,config,project,entity,backfilled=True)
    for row in rows:
        if row['epoch'] >= run.step:
            run.log(flatten_metrics(row),step=row['epoch'])
    set_summary(run,run_dir,rows,progress)
    run.finish()
    saved=json.loads(pointer.read_text())
    saved.update(synced_through_epoch=rows[-1]['epoch'],training_status=progress['status'],backfilled=True)
    pointer.write_text(json.dumps(saved,indent=2)+'\n')
    print('Synced all historical epochs:',saved['url'])


def main(argv=None):
    parser=argparse.ArgumentParser(description='Upload a finished run\'s local metrics.jsonl to W&B.')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--project',default=DEFAULT_PROJECT)
    parser.add_argument('--entity')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    backfill(args.run_dir,args.project,args.entity,args.dry_run)
