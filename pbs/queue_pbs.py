"""
Script to add new PBS jobs, creates PBS scripts for the specified datasets and methods.

Checks available output files and creates PBS scripts for the missing ones.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>

Usage:
```
python queue_pbs.py <datasets> [--shuffle] [-m <methods>] [-t <train_ratio>] [-s <settings>] [-d]
```
"""

import argparse
import os
from pathlib import Path

SETTINGS = {
    "general": {
        "homedir": "/storage/brno2/home/$USER/",
        "setup": "$HOMEDIR/jobs/setup.sh",
        "teardown": "$HOMEDIR/jobs/teardown.sh",
        "script_dir": "$HOMEDIR/jobs/scripts/",
        "output_dir": "$HOMEDIR/outputs/",
    },
    "resources": {
        "Short": [
            "select=1:ncpus=2:mem=32gb:scratch_local=40gb",
            "walltime=4:00:00",
        ],
        "DeepLog": [
            "select=1:ncpus=1:mem=32gb:ngpus=1:gpu_mem=10gb:scratch_local=40gb",
            "walltime=16:00:00",
        ],
        "LogAnomaly": [
            "select=1:ncpus=1:mem=40gb:ngpus=1:gpu_mem=10gb:scratch_local=40gb",
            "walltime=24:00:00",
        ],
        "LogBERT": [
            "select=1:ncpus=2:mem=32gb:ngpus=1:gpu_mem=10gb:gpu_cap=compute_70:scratch_local=64gb",
            "walltime=22:00:00",
        ],
        "LogRobust": [
            "select=1:ncpus=1:mem=32gb:ngpus=1:gpu_mem=10gb:scratch_local=40gb",
            "walltime=20:00:00",
        ],
        "NeuralLog": [
            "select=1:ncpus=1:mem=64gb:ngpus=1:gpu_mem=16gb:scratch_local=40gb",
            "walltime=16:00:00",
        ],
    },
    "outputs": {
        "Short": ["SVM", "PCA", "SemPCA", "LogCluster"],
        "DeepLog": ["DeepLog"],
        "LogAnomaly": ["LogAnomaly"],
        "LogBERT": ["LogBERT"],
        "LogRobust": ["LogRobust"],
        "NeuralLog": ["NeuralLog"],
    },
}


def get_script_name(dataset: str, method: str, shuffle: bool, ratio: float) -> str:
    return f"{dataset}-{method}-{'rng' if shuffle else 'seq'}-{ratio}.pbs"


def resolve_path(settings: dict, path: str) -> Path:
    path = path.replace("$HOMEDIR", settings["general"]["homedir"])
    if "$USER" in path:
        username = os.environ["USER"]
        path = path.replace("$USER", username)
    return Path(path).expanduser().resolve()


def get_script_dir(settings: dict) -> Path:
    script_dir = settings["general"]["script_dir"]
    return resolve_path(settings, script_dir)


def get_output_dir(settings: dict) -> Path:
    output_dir = settings["general"]["output_dir"]
    return resolve_path(settings, output_dir)


def outputs_exist(method_out_dir: Path) -> bool:
    if method_out_dir.exists():
        metrics = list(method_out_dir.glob("metrics_0.?.csv"))
        if len(metrics) < 10:
            print(f"Found {len(metrics)}/10 metrics files in {method_out_dir}.")
            return False
        else:
            print(f"Found all metrics files in {method_out_dir}.")
            return True
    return False


def create_pbs_script(
    dataset: str, method: str, shuffle: bool, ratio: float, settings: dict
):
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
    opts = [
        ("--shuffle" if shuffle else ""),
        f"--train_ratio {ratio}",
    ]
    opts = " ".join(filter(None, opts))
    method_template = method_template.replace("$OPTS", opts)
    script_lines.append(method_template)

    # Teardown
    teardown_script = settings["general"]["teardown"]
    script_lines.append(f"source {teardown_script}\n")

    # Write the script to a file
    script_name = get_script_name(dataset, method, shuffle, ratio)
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
        choices=[
            "HDFS",
            "HDFSFixed",
            "HDFSLogHub",
            "BGL40",
            "BGL120",
            "BGLComponent",
            "TBird",
        ],
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
    parser.add_argument(
        "-m",
        "--methods",
        type=str,
        nargs="+",
        help="Methods to use. If not specified, all methods will be used.",
    )
    parser.add_argument(
        "-t",
        "--train_ratio",
        type=float,
        help="Train ratio to use.",
        default=[0.5],
        nargs="+",
    )
    parser.add_argument(
        "-s",
        "--settings",
        type=str,
        help="Settings file to use.",
        default=str(Path(__file__).parent / "settings.toml"),
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="If set, the script will not submit jobs.",
    )
    args = parser.parse_args()

    settings = SETTINGS

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

    output_dir = get_output_dir(settings)

    # Plan what jobs to run
    to_run = {}
    dataset_ratio = ((d, t) for t in args.train_ratio for d in set(args.datasets))
    for dataset, train_ratio in dataset_ratio:
        pattern = f"{dataset}{'Shuffled' if args.shuffle else ''}*_{train_ratio}"
        globbed = list(output_dir.glob(pattern))
        if not args.shuffle:
            globbed = [d for d in globbed if "Shuffled" not in str(d)]
        if len(globbed) > 1:
            print(f"Matched {globbed} for {pattern}. Skipping.")
            continue
        if len(globbed) == 0:
            to_run[dataset, train_ratio] = {
                method: False for method in selected_methods
            }
            continue
        dataset_dir = globbed[0]

        to_run[dataset, train_ratio] = {}
        for method in selected_methods:
            method_outputs = settings["outputs"][method]
            all_exist = all(outputs_exist(dataset_dir / out) for out in method_outputs)
            to_run[dataset, train_ratio][method] = all_exist

    for i, ((dataset, ratio), methods) in enumerate(
        sorted(to_run.items(), key=lambda x: (x[0][1], x[0][0]))
    ):
        if i == 0:
            print(f"{'':20s} {'':5s} " + " ".join(f"{m:15s}" for m in methods))
        print(
            f"{dataset:20s} {ratio:5.2f} "
            + " ".join(f"{('Run' if not m else '-'):15s}" for m in methods.values())
        )
    try:
        if input("Proceeed? [y/N]: ").lower() != "y":
            exit(0)
    except KeyboardInterrupt:
        print()
        exit(0)

    # Create and submit jobs
    for (dataset, train_ratio), methods in to_run.items():
        for method, exists in methods.items():
            if exists:
                continue

            script_path = script_dir / get_script_name(
                dataset, method, args.shuffle, train_ratio
            )
            if script_path.exists():
                print(f"Script {script_path} already exists. Skipping.")
            else:
                create_pbs_script(dataset, method, args.shuffle, train_ratio, settings)

            print("Submitting job for", dataset, method)
            if not args.dry_run:
                os.system(f"qsub {script_path}")
            else:
                print(f"Dry run: qsub {script_path} (not actually submitting)")


if __name__ == "__main__":
    main()
