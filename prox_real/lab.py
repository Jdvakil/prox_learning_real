"""Lab PC helpers behind utils.py: setup check, transfer archive and a robot smoke test."""
import argparse
import importlib
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"act_test_20": ["policy_best.ckpt", "dataset_stats.npz", "training_config.json", "image_metadata.json"],
          "pc_act_test_20": ["policy_best.ckpt", "prox_encoder_best.pt", "policy_pair.json", "guard_baseline.npz"],
          "pick_act": ["policy_best.ckpt", "dataset_stats.npz", "pick_policy.json"],
          "pick_pcact": ["policy_best.ckpt", "prox_encoder_best.pt", "prox_config.json", "dataset_stats.npz", "pick_policy.json"]}


def check(argv=None):
    """Check local dependencies and model files without opening cameras or connecting to the robot."""
    p=argparse.ArgumentParser(description=check.__doc__)
    p.add_argument("--offline-only",action="store_true",help="skip robot and camera driver checks")
    args=p.parse_args(argv)
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
    for model,names in MODELS.items():
        for name in names:
            path=ROOT/"models"/model/name
            ok=path.is_file()
            if ok and path.suffix in {".ckpt",".pt"}:
                with path.open('rb') as f:ok=not f.read(128).startswith(b'version https://git-lfs.github.com/spec/')
            print(f"{'OK' if ok else 'MISSING'} {path.relative_to(ROOT)}")
            if not ok:failures.append(name)
    if failures:sys.exit(1)
    print("Setup checks passed. No hardware connections were made.")


def bundle(argv=None):
    """Make a transferable code + model + Franky-source archive; excludes recordings and secrets."""
    p=argparse.ArgumentParser(description=bundle.__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"dist/prox_learning_real_lab.tar.gz")
    args=p.parse_args(argv)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    files=[ROOT/name for name in ["data.py","model.py","utils.py","README.md","pick_pose.json"]]+list(ROOT.glob("requirements-*.txt"))
    for directory in ["prox_real","act_model","models","sim_assets","submodules/franky"]:
        files += [path for path in (ROOT/directory).rglob("*") if path.is_file()
                  and not any(part in {".git","__pycache__",".pytest_cache","build","dist"} for part in path.relative_to(ROOT).parts)
                  and path.suffix not in {".pyc",".so"}]
    # Do not ship symlinks which could resolve outside this standalone tree.
    with tarfile.open(args.output,"w:gz",compresslevel=3,dereference=False) as tar:
        for path in sorted(set(files)):
            if not path.is_symlink():tar.add(path,arcname=str(Path("prox_learning_real")/path.relative_to(ROOT)),recursive=False)
    print(f"Wrote {args.output} ({args.output.stat().st_size/1024**2:.1f} MiB)")


def robot_test(argv=None):
    """Move the end effector 20 cm along its own x axis at 5% of the velocity/acceleration/jerk limits."""
    p=argparse.ArgumentParser(description=robot_test.__doc__)
    p.add_argument("--host",default="172.16.0.2")
    p.add_argument("--distance",type=float,default=0.2,help="m along the end-effector x axis")
    args=p.parse_args(argv)
    from franky import Affine, CartesianMotion, ReferenceType, Robot

    robot=Robot(args.host)
    robot.relative_dynamics_factor=0.05
    robot.move(CartesianMotion(Affine([args.distance,0.0,0.0]),ReferenceType.Relative))
