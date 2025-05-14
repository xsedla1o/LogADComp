"""
Script to manage collected artefacts.
Can be used to clear empty directories or create a view of the artefacts using symlinks.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>

Usage:
```
python archeologist.py clear-empty
python archeologist.py view artefacts_method_view MM/DD_RR/AA
```
"""

import argparse
import os
import shutil
from pathlib import Path

import tomli


def exists_and_not_empty(file_path: str) -> bool:
    return (
        os.path.exists(file_path)
        and os.path.isfile(file_path)
        and os.path.getsize(file_path) > 0
    )


def subtree_has_files(path: str) -> bool:
    for root, dir_names, files in os.walk(path):
        if files:
            return True
        for dir_name in dir_names:
            if subtree_has_files(os.path.join(root, dir_name)):
                return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to manage collected artefacts."
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )
    subparsers.add_parser("clear-empty", help="Clear empty directories")
    view_parser = subparsers.add_parser(
        "view", help="Create a view of the artefacts using symlinks"
    )
    view_parser.add_argument("to", type=str, help="Directory to construct view in")
    view_parser.add_argument(
        "view_pattern",
        type=str,
        help="Pattern to construct view. Use DD=dataset, MM=method, RR=ratio, AA=artefacts. Default: DD_RR/MM/AA",
        default="DD_RR/MM/AA",
        nargs="?",
    )
    args = parser.parse_args()

    with open("paths.toml", "rb") as f:
        dir_config = tomli.load(f)
    base_path = Path(dir_config["outputs"]) / ".."
    artefact_path = (base_path / "artefacts").resolve()

    if args.command == "clear-empty":
        for subdir in artefact_path.iterdir():
            if subdir.is_dir() and not subtree_has_files(subdir):
                print(f"Removing empty directory: {subdir}")
                shutil.rmtree(subdir)

    elif args.command == "view":
        view_pattern = args.view_pattern
        view_dst = Path(args.to)
        if not view_dst.exists():
            view_dst.mkdir(parents=True, exist_ok=True)

        for art_dir in artefact_path.iterdir():
            if not art_dir.is_dir():
                continue

            dataset_dirs = [d for d in art_dir.iterdir() if d.is_dir()]
            if not dataset_dirs:
                continue
            if len(dataset_dirs) > 1:
                print(f"Multiple datasets found in {art_dir}. Skipping.")
                continue
            dataset_dir = dataset_dirs[0]
            d_name, train_ratio = dataset_dir.name.rsplit("_", maxsplit=1)

            method_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            if not method_dirs:
                continue
            if len(method_dirs) > 1:
                print(f"Multiple methods found in {dataset_dir}. Skipping.")
                continue
            method_dir = method_dirs[0]
            m_name = method_dir.name

            # make new dst
            with_dset = view_pattern.replace("DD", d_name)
            with_meth = with_dset.replace("MM", m_name)
            with_ratio = with_meth.replace("RR", train_ratio)
            with_art = with_ratio.replace("AA", art_dir.name)
            new_dst = view_dst / with_art

            if not new_dst.exists():
                new_dst.parent.mkdir(parents=True, exist_ok=True)
                new_dst.symlink_to(method_dir, target_is_directory=True)
            else:
                print(f"Directory {new_dst} already exists. Skipping.")
