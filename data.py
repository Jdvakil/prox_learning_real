"""Record and inspect robot data. Run `python data.py` for the command list."""
import sys

from prox_real.cli import run

COMMANDS = {
    "cameras": ("cameras:main", "list RealSense cameras, save snapshots, or --live view / --record"),
    "record": ("record_dataset:main", "move through POSITIONS and log robot, 40 sensors and cameras"),
    "waypoints": ("record_act_waypoints:main", "record noisy A -> B episodes for ACT (--episodes N)"),
    "teach": ("kinesthetic:teach_cli", "kinesthetic teaching: move the floating arm by hand"),
    "replay": ("kinesthetic:replay_cli", "replay taught demos on the stiff tracker and log rows"),
    "dashboard": ("dashboard:main", "offline HTML dashboard of recorded rows -> viz/"),
    "analyze": ("analysis:main", "audit an A -> B batch: metrics, plots, HTML report"),
}

if __name__ == "__main__":
    sys.exit(run("data.py", __doc__.split(".")[0] + ".", COMMANDS))
