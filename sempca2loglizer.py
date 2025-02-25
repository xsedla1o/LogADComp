"""
LineId,Date,Time,Pid,Level,Component,Content,EventId,EventTemplate

"""

import csv
import sys

sys.path.extend(["sempca", "sempca/preprocessing"])

from sempca.preprocessing.dataloader.HDFSLoader import HDFSLoader

from sempca.CONSTANTS import *
import argparse


def generate_inputs_and_labels(insts, label2idx):
    inputs = []
    labels = np.zeros(len(insts))
    for idx, inst in enumerate(insts):
        inputs.append([int(x) for x in inst.sequence])
        label = int(label2idx[inst.label])
        labels[idx] = label
    return inputs, labels


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", default="HDFS", type=str, help="Target dataset. Default HDFS"
    )
    argparser.add_argument(
        "--parser",
        default="IBM",
        type=str,
        help="Select parser, please see parser list for detail. Default IBM.",
    )
    argparser.add_argument(
        "--max_dist",
        type=float,
        default=0.3,
        help="Max Distance parameter in LogClustering.",
    )
    argparser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=None,
        help="Anomaly Threshold parameter in LogClustering.",
    )
    argparser.add_argument(
        "--save_results", type=bool, default=False, help="Whether to save results."
    )
    args, extra_args = argparser.parse_known_args()

    dataset = args.dataset
    parser = args.parser
    max_dist = args.max_dist
    anomaly_threshold = args.anomaly_threshold
    save_results = args.save_results

    dataloader = HDFSLoader(
        in_file=os.path.join(PROJECT_ROOT, "datasets/HDFS/HDFS.log"),
        semantic_repr_func=None,
    )

    parser_config = os.path.join(PROJECT_ROOT, "conf/HDFS.ini")
    parser_persistence = os.path.join(PROJECT_ROOT, "datasets/HDFS/persistences")
    csv_out = "data/HDFS.csv"
    encode = "utf-8"

    dataloader.parse_by_IBM(
        config_file=parser_config, persistence_folder=parser_persistence, encode=encode
    )

    with (
        open(dataloader.in_file, "r", encoding=encode) as in_f,
        open(csv_out, "w", encoding=encode) as out_f,
    ):
        writer = csv.writer(
            out_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(
            (
                "LineId",
                "Date",
                "Time",
                "Pid",
                "Level",
                "Component",
                "Content",
                "EventId",
                "EventTemplate",
            )
        )

        for line_id, line in tqdm(enumerate(in_f)):
            tokens = line.strip().split()
            date, time, pid, level, component, *content_tokens = tokens

            if level not in ["INFO", "WARN", "ERROR"]:
                continue  # Skip malformed lines

            content = " ".join(content_tokens)
            component = component.split(":")[0]
            event_id = dataloader.log2temp[line_id]
            event_template = dataloader.templates[event_id]

            writer.writerow(
                (
                    line_id,
                    date,
                    time,
                    pid,
                    level,
                    component,
                    content,
                    event_id,
                    event_template,
                )
            )
