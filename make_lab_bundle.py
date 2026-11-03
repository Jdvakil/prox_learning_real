"""Make a transferable code + model + Franky-source archive; excludes recordings/secrets."""
import argparse
from pathlib import Path
import tarfile


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=root/"dist/prox_learning_real_lab.tar.gz")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    files=[root/name for name in [
        'record_act_waypoints.py','record_dataset.py','kinesthetic.py','cameras.py',
        'sensor_layout.py','eval_act_real.py','export_act_model.py','check_lab_setup.py',
        'make_lab_bundle.py']]+list(root.glob('requirements-*.txt'))
    files += [root/'README.md',root/'LAB_SETUP.txt']
    for directory in ['act_model','models','submodules/franky']:
        files += [path for path in (root/directory).rglob('*') if path.is_file()
                  and not any(part in {'.git','__pycache__','.pytest_cache','build','dist'} for part in path.relative_to(root).parts)
                  and path.suffix not in {'.pyc','.so'}]
    # Do not ship symlinks which could resolve outside this standalone tree.
    with tarfile.open(args.output,'w:gz',compresslevel=3,dereference=False) as tar:
        for path in sorted(set(files)):
            if not path.is_symlink():tar.add(path,arcname=str(Path('prox_learning_real')/path.relative_to(root)),recursive=False)
    print(f"Wrote {args.output} ({args.output.stat().st_size/1024**2:.1f} MiB)")


if __name__=="__main__":main()
