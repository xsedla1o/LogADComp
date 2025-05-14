"""
Script to process datasets and collect event distributions.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import pandas as pd
import tomli

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataloader import dataloaders
from sempca.preprocessing import DataPaths
from utils import seed_everything

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None


def process_dataset(dir_config, d_name, shuffled: bool = False):
    dataset_dir = dir_config["datasets"]
    suffix = "Shuffled" if args.shuffle else ""
    dataset_srcdir = f"{dataset_dir}/{d_name}"
    dataset_processed_dir = f"{dataset_srcdir}"

    for d in [dataset_srcdir, dataset_processed_dir]:
        os.makedirs(d, exist_ok=True)

    config_dict = {
        "dataset_dir": dataset_dir,
        "dataset": f"{dataset_srcdir}/{d_name}.log",
        "labels": f"{dataset_srcdir}/anomaly_label.csv",
        "processed_labels": f"{dataset_processed_dir}/label.csv",
        "embeddings": f"{dataset_dir}/glove.6B.300d.txt",
        "loglizer_seqs": f"{dataset_processed_dir}/{d_name}.seqs.csv",
        "word_vec_npz": f"{dataset_processed_dir}/{d_name}.word_vec.npz",
        "ecv_npz": f"{dataset_processed_dir}/{d_name}.ecv.npz",
        "t_seq_npz": f"{dataset_processed_dir}/{d_name}.t_seq.npz",
        "embed_seq_npz": f"{dataset_processed_dir}/{d_name}.embed_seq.npz",
        "random_indices": f"{dataset_processed_dir}/{d_name}.random_indices.out",
    }

    # get abspath of this script
    root_dir = Path(__file__).parent.parent.parent
    paths = DataPaths(
        dataset_name=d_name,
        project_root=root_dir,
        datasets_dir=config_dict["dataset_dir"],
        label_file=config_dict["processed_labels"],
    )
    output_dir = Path(root_dir) / "plots" / "event_distribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{d_name}{suffix}"

    dataloader = dataloaders[d_name](config_dict, paths)
    dataloader.get()
    xs, ys = dataloader.get_ecv_representation()

    event_counts = []
    event_counts_normal = []
    event_counts_anomaly = []

    for offset in range(0, 10, 2):
        offset /= 10
        seed_everything()
        (x_train, y_train), (x_val, y_val), _ = dataloader.split(
            xs,
            ys,
            train_ratio=0.1,
            val_ratio=0.1,
            offset=offset,
            shuffle=shuffled,
        )

        event_counts.append(x_train.sum(axis=0))
        event_counts.append(x_val.sum(axis=0))

        event_counts_normal.append(x_train[y_train == 0].sum(axis=0))
        event_counts_normal.append(x_val[y_val == 0].sum(axis=0))

        event_counts_anomaly.append(x_train[y_train == 1].sum(axis=0))
        event_counts_anomaly.append(x_val[y_val == 1].sum(axis=0))

        del x_train, y_train, x_val, y_val
        gc.collect()

    os.makedirs(f"{output_dir}/{out_name}", exist_ok=True)
    event_counts = pd.DataFrame(event_counts)
    event_counts.to_csv(f"{output_dir}/{out_name}/all.csv", index=False)

    event_counts_normal = pd.DataFrame(event_counts_normal)
    event_counts_normal.to_csv(f"{output_dir}/{out_name}/normal.csv", index=False)

    event_counts_anomaly = pd.DataFrame(event_counts_anomaly)
    event_counts_anomaly.to_csv(f"{output_dir}/{out_name}/anomaly.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to use for anomaly detection, choices: " + ", ".join(dataloaders),
        choices=list(dataloaders.keys()),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before splitting",
        default=False,
    )
    args = parser.parse_args()
    split_offsets = list(range(0, 10))

    with open("paths.toml", "rb") as f:
        dir_conf = tomli.load(f)

    process_dataset(dir_conf, args.dataset, args.shuffle)
