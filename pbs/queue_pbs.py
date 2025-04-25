"""
Script to add new PBS jobs.
"""

import argparse
import os
import sys
from pathlib import Path

import tomli

sys.path.append(str(Path(__file__).parent.parent))

from dataloader import dataloaders


def get_script_name(dataset: str, method: str, shuffle: bool) -> str:
    return f"{dataset}-{method}-{'rng' if shuffle else 'seq'}.pbs"


def get_script_dir(settings: dict) -> Path:
    script_dir = settings["general"]["script_dir"]
    script_dir = Path(script_dir.replace("$HOMEDIR", settings["general"]["homedir"]))
    return script_dir.expanduser().resolve()


def create_pbs_script(dataset: str, method: str, shuffle: bool, settings: dict):
    """
    Create a PBS script for the given dataset and method.

    Args:
        dataset (str): The name of the dataset.
        method (str): The method to use for the job.
        shuffle (bool): Whether to shuffle the dataset.
        settings (dict): Settings of resources and paths.
    """
    # PBS header
    script_lines = [
        "#!/bin/bash",
        f"#PBS -N {dataset}-{method}-{'rng' if shuffle else 'seq'}",
    ]
    for value in settings["resources"][method]:
        script_lines.append(f"#PBS -l {value}")
    script_lines.append("#PBS -j oe\n")

    # Define variables
    homedir = settings["general"]["homedir"]
    script_lines.append(f'HOMEDIR="{homedir}"\n')

    # Setup environment
    setup_script = settings["general"]["setup"]
    script_lines.append(f"source {setup_script}\n")

    # Method template
    with open(Path(__file__).parent / "templates" / f"{method}.sh", "r") as f:
        method_template = f.read()
    method_template = method_template.replace("$DATASET", dataset)
    opts = "--shuffle" if shuffle else ""
    method_template = method_template.replace("$OPTS", opts)
    script_lines.append(method_template)

    # Teardown
    teardown_script = settings["general"]["teardown"]
    script_lines.append(f"source {teardown_script}\n")

    # Write the script to a file
    script_name = f"{dataset}-{method}-{'rng' if shuffle else 'seq'}.pbs"
    script_path = get_script_dir(settings) / script_name
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines))
    print("Script created:", script_path)


def main():
    parser = argparse.ArgumentParser(description="Add new PBS jobs.")
    parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        help="Datasets on which to run the jobs.",
        choices=list(dataloaders.keys()),
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        help="Methods to use. If not specified, all methods will be used.",
    )
    parser.add_argument(
        "--settings",
        "-s",
        type=str,
        help="Settings file to use.",
        default=str(Path(__file__).parent / "settings.toml"),
    )
    args = parser.parse_args()

    with open(args.settings, "rb") as f:
        settings = tomli.load(f)

    # Ensure script directory exists
    script_dir = get_script_dir(settings)
    script_dir.mkdir(parents=True, exist_ok=True)

    if args.methods is None:
        selected_methods = settings["resources"].keys()
    else:
        for method in args.methods:
            if method not in settings["resources"]:
                raise ValueError(f"Method {method} not found in settings.")
        selected_methods = args.methods

    for dataset in set(args.datasets):
        for method in selected_methods:
            script_path = script_dir / get_script_name(dataset, method, args.shuffle)
            if script_path.exists():
                print(f"Script {script_path} already exists. Skipping.")
            else:
                create_pbs_script(dataset, method, args.shuffle, settings)

            print("Submitting job for", dataset, method)
            os.system(f"qsub {script_path}")


if __name__ == "__main__":
    main()
