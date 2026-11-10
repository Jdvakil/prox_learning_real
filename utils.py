"""Lab PC setup, transfer and hardware checks. Run `python utils.py` for the command list."""
import sys

from prox_real.cli import run

COMMANDS = {
    "check": ("lab:check", "check Python packages and the deployed models (no hardware)"),
    "bundle": ("lab:bundle", "write dist/prox_learning_real_lab.tar.gz for the lab PC"),
    "robot-test": ("lab:robot_test", "move the end effector 20 cm forward at 5% speed"),
    "sim": ("sim.twin:main", "rehearse teach / replay on the MuJoCo bench (needs mujoco)"),
}

if __name__ == "__main__":
    sys.exit(run("utils.py", __doc__.split(".")[0] + ".", COMMANDS))
