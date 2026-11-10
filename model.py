"""Train, export and evaluate ACT / PC-ACT policies. Run `python model.py` for the command list."""
import sys

from prox_real.cli import run

COMMANDS = {
    "train": ("train_pc_act:main", "train PC-ACT on recorded episodes (--detach to run in the background)"),
    "eval": ("evaluate:main", "evaluate a policy: eval act|pc-act [--check-model | --offline-row ROW | live]"),
    "export": ("export_model:main", "copy a trained run into models/: export act|pc-act --source RUN"),
    "wandb-sync": ("pc_act_wandb:main", "upload a finished run's local metrics to W&B"),
}

if __name__ == "__main__":
    sys.exit(run("model.py", __doc__.split(".")[0] + ".", COMMANDS))
