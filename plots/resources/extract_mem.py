"""
Extract memory usage from orchestrator logs.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
from pathlib import Path
from typing import Union

import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

orchestrator_prints = [
    "Base usage",
    "Transformed usage",
    "Pre GC usage",
    "Post GC usage",
    "Post train params usage",
    "Memory usage",
    "Post params usage",
]

multipliers = {
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}


def parse_memstr(mem_str: str) -> float:
    """
    Parse a memory string into bytes.
    e.g. "1.5GB" -> 1.5 * 1e9
    """
    if mem_str[-1] == "B":
        mem_str = mem_str[:-1]
    mult = multipliers.get(mem_str[-1], 1)
    mem_str = mem_str[:-1] if str.isdigit(mem_str[-2]) else mem_str[:-2]
    return float(mem_str) * mult


def get_mem_usage(filepath: Union[str, Path]):
    """
    Get the memory usage from a log file.
    """
    usage_kind = []
    rss = []
    vms = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if any(line.startswith(p) for p in orchestrator_prints):
                kind = line.split(" usage")[0]
                usage_kind.append(kind)
                mem_part = line.split("'rss': ", maxsplit=1)[-1]
                parts = mem_part.split("'")
                mem_str = parts[1]
                rss.append(parse_memstr(mem_str))

                mem_part = line.split("'vms': ", maxsplit=1)[-1]
                parts = mem_part.split("'")
                mem_str = parts[1]
                vms.append(parse_memstr(mem_str))

            if line.startswith("DataPaths") and rss:
                yield pd.DataFrame({"note": usage_kind, "rss": rss, "vms": vms})
                usage_kind, rss, vms = [], [], []

    if rss:
        yield pd.DataFrame({"note": usage_kind, "rss": rss, "vms": vms})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="Log files to plot",
        nargs="+",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="Output directory for extracted times",
    )
    args = parser.parse_args()
    out = Path(args.out_dir)
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()

    short_names = ["SVM", "PCA", "SemPCA", "LogCluster"]

    for file in args.file:
        fp = Path(file)
        if not fp.exists():
            print(f"File {fp} does not exist")
            continue

        name = fp.stem.split("-")[1]
        i = 0
        for mem_history in get_mem_usage(fp):
            if mem_history.shape[0] == 0:
                continue

            stem = short_names[i] if name == "Short" else name
            out_path = out / f"{stem}_{i}.csv"
            if out_path.exists():
                prev_history = pd.read_csv(out_path)
                mem_history = pd.concat([prev_history, mem_history])
                mem_history = mem_history.drop_duplicates()
            i += 1
            mem_history.to_csv(out_path, index=False)
