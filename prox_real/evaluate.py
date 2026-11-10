"""Pick the ACT or PC-ACT evaluator: `python model.py eval act|pc-act [options]`."""
import sys

KINDS = {"act": "eval_act_real", "pc-act": "eval_pc_act_real"}


def main(argv):
    if not argv or argv[0] not in KINDS:
        raise SystemExit("usage: python model.py eval {act,pc-act} [options]   (add -h for the evaluator's options)")
    import importlib

    sys.argv[0] = f"model.py eval {argv[0]}"
    return importlib.import_module(f"prox_real.{KINDS[argv[0]]}").main(argv[1:])
