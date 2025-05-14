"""
Show memory usage of the models in a formatted LaTeX table.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return f"{bytes:6.2f}B"
    elif abs(bytes) < 1e6:
        return f"{round(bytes / 1e3, 2):6.2f}kB"
    elif abs(bytes) < 1e9:
        return f"{round(bytes / 1e6, 2):6.2f}MB"
    else:
        return f"{round(bytes / 1e9, 2):6.2f}GB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory with extracted time druation csv files",
        nargs="+",
    )

    args = parser.parse_args()
    order = {"B": 0, "H": 1, "T": 2}
    labels = {
        "B": "BGL",
        "H": "HDFS",
        "T": "TBird",
    }
    ds_lengths = {
        "B": 4747963,
        "H": 11197705,
        "T": 20000000,
    }
    recs = []

    for data_dir in args.data_dir:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            continue

        first = data_dir.stem.split("_")[0][0]
        ds_label = labels.get(first, first)

        contents = {}
        for file in data_dir.glob("*.csv"):
            df = pd.read_csv(file)
            df = df[~df["note"].str.contains("GC")]
            df = df[
                ~((df["note"].eq("Post params").cumsum() < 1) & df["note"].eq("Memory"))
            ]
            df.reset_index(drop=True, inplace=True)

            name = file.stem.split("_")[0]
            contents[name] = df

        # Plot memory usage
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, df in contents.items():
            (df["rss"] / 1e9).plot(ax=ax, label=name)

        df_sample = next(iter(contents.values()))  # Pick any one dataframe
        for note in ["Transformed", "Post params", "Memory"]:
            if note in df_sample["note"].values:
                idx = df_sample[df_sample["note"] == note].index[0]
                ax.axvline(x=idx, color="blue", linestyle="--", alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_xticks(range(0, len(df)))
        ax.set_ylabel("RSS Memory usage [GB]")
        ax.legend(loc="upper right")
        plt.savefig(data_dir / "memory_usage.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

        for name, df in contents.items():
            summary = {
                "Dataset": ds_label,
                "Name": name,
                "Data Loaded": df.loc[df["note"] == "Transformed", "rss"].values[0],
                "Operation": df.loc[df["note"] == "Memory", "rss"].values[0],
            }
            summary["Clean"] = summary["Operation"] - summary["Data Loaded"]
            summary["Peak"] = df["rss"].max()
            summary["Per Line"] = (
                summary["Operation"] - df.loc[df["note"] == "Base", "rss"].values[0]
            ) / ds_lengths[first]
            recs.append(summary)

    sum_df = pd.DataFrame(recs).set_index(["Name", "Dataset"]).sort_index()
    sum_df = sum_df.groupby("Name", group_keys=False).apply(lambda x: x.sort_index())
    sum_df = sum_df.loc[
        sum_df.groupby("Name")["Clean"].sum().sort_values(ascending=True).index
    ]
    ltx_str: str = sum_df.to_latex(
        index=True,
        formatters={
            "Data Loaded": lambda x: format_bytes(x),
            "Operation": lambda x: format_bytes(x),
            "Clean": lambda x: format_bytes(x),
            "Peak": lambda x: format_bytes(x),
            "Per Line": lambda x: format_bytes(x),
        },
    )
    ltx_str = ltx_str.replace("\\multirow[t]", "\\multirow")
    ltx_str = ltx_str.replace("\\cline{1-7}", "\\cline{2-7}")
    ltx_str = ltx_str.replace("\\cline{2-7}\n\\bottomrule", "\\bottomrule")

    print(ltx_str)
