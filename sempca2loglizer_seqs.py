import argparse
import csv
import os

from sempca.const import PROJECT_ROOT
from sempca.preprocessing import HDFSLoader, BGLLoader, DataPaths

settings = {
    "HDFS": {
        "dataloader": HDFSLoader,
        "in_file": os.path.join(PROJECT_ROOT, "datasets/HDFS/HDFS.log"),
        "semantic_repr_func": None,
        "parser_config": os.path.join(PROJECT_ROOT, "conf/HDFS.ini"),
        "parser_persistence": os.path.join(PROJECT_ROOT, "datasets/HDFS/persistences"),
        "csv_out": "data/HDFS.seqs.csv",
    },
    "BGL": {
        "dataloader": BGLLoader,
        "in_file": os.path.join(PROJECT_ROOT, "datasets/BGL/BGL.log"),
        "semantic_repr_func": None,
        "parser_config": os.path.join(PROJECT_ROOT, "conf/BGL.ini"),
        "parser_persistence": os.path.join(PROJECT_ROOT, "datasets/BGL/persistences"),
        "csv_out": "data/BGL.seqs.csv",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        default="HDFS",
        choices=["HDFS", "BGL"],
        type=str,
        help="Target dataset. Default HDFS",
    )
    args = parser.parse_args()

    dataset = args.dataset
    s = settings[dataset]

    paths = DataPaths(dataset_name=dataset)

    dataloader = s["dataloader"](
        paths=paths,
        semantic_repr_func=None,
    )

    dataloader.parse_by_drain(
        core_jobs=os.cpu_count() // 2,
    )

    # Drop malformed template
    m_id = None
    for t_id, template in dataloader.templates.items():
        if template == "such file or directory":
            m_id = t_id
            break

    with open(s["csv_out"], "w") as out_f:
        writer = csv.writer(
            out_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(("BlockId", "Label", "EventSequence"))

        for block, sequence in dataloader.block2eventseq.items():
            seq = " ".join(str(x) for x in sequence if x != m_id)
            label_id = dataloader.label2id[dataloader.block2label[block]]
            writer.writerow((block, label_id, seq))
