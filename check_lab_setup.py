"""Check local dependencies without opening cameras or connecting to the robot."""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only",action="store_true",help="skip robot and camera driver checks")
    args=p.parse_args()
    required=["numpy","h5py","cv2","torch","torchvision","imageio_ffmpeg"]
    if not args.offline_only:required += ["pyrealsense2","franky"]
    failures=[]
    for name in required:
        try:
            module=importlib.import_module(name)
            if name=="franky" and not hasattr(module,"JointImpedanceTracker"):
                raise RuntimeError("installed Franky lacks JointImpedanceTracker; build the included source")
            print(f"OK {name}: {getattr(module,'__version__','importable')}")
        except Exception as exc:
            failures.append(name);print(f"MISSING {name}: {exc}")
    root=Path(__file__).resolve().parent
    for name in ["policy_best.ckpt","dataset_stats.npz","training_config.json","image_metadata.json"]:
        path=root/"models/act_test_20"/name
        ok=path.is_file()
        if ok and name.endswith('.ckpt'):
            with path.open('rb') as f:ok=not f.read(128).startswith(b'version https://git-lfs.github.com/spec/')
        print(f"{'OK' if ok else 'MISSING'} {path.relative_to(root)}")
        if not ok:failures.append(name)
    if failures:sys.exit(1)
    print("Setup checks passed. No hardware connections were made.")


if __name__=="__main__":main()
