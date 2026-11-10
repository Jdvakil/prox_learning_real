"""Audit a recorded A -> B batch: metrics, plots and an HTML report.

    python data.py analyze                                         # recordings/act_test_20 -> act_20_analysis/
    python data.py analyze recordings/<batch> --out <analysis dir>
"""
import argparse
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(source, out):
    from . import analyze

    out.mkdir(parents=True, exist_ok=True)
    analyze.SOURCE, analyze.OUT = Path(source), Path(out)
    analyze.main()
    for script in ("plot_analysis.py", "make_report.py"):
        runpy.run_path(str(HERE / script), init_globals={"OUT": Path(out)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, nargs="?", default=ROOT / "recordings/act_test_20")
    parser.add_argument("--out", type=Path, default=ROOT / "act_20_analysis")
    args = parser.parse_args(argv)
    run(args.source, args.out)
