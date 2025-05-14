"""
Show runtime durations of the models in a formatted LaTeX table.

Author: Ondřej Sedláček <xsedla1o@stud.fit.vutbr.cz>
"""

import argparse
from pathlib import Path

import pandas as pd


def format_time(seconds, show_h=False):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if show_h or hours != 0:
        return f"{hours:02}:{minutes:02}:{seconds:05.2f}"
    else:
        return f"{minutes:02}:{seconds:05.2f}"


def extract_model_data(file):
    """Extract model data from the CSV file."""
    df = pd.read_csv(file)
    optimize = df.loc[0, "Optimize"]

    df = df.drop(columns=["Optimize"])
    # Drop empty rows
    df = df[(df.T != 0).any()]

    # Get the model name
    model_name = file.stem.split("_")[0]
    # Get the GPU model
    gpu_model = df["GPU"].iloc[0]
    print(df["GPU"].unique())
    gpu_model = gpu_model if gpu_model != 0 else None

    # Calculate means and stds
    fit_mean = df["Fit"].mean()
    fit_std = df["Fit"].std()
    predict_mean = df["Predict"].mean()
    predict_std = df["Predict"].std()

    # Format times
    optimize_formatted = format_time(optimize, show_h=True)
    fit_mean_formatted = format_time(fit_mean, show_h=True)
    fit_std_formatted = format_time(fit_std)
    predict_mean_formatted = format_time(predict_mean)
    predict_std_formatted = format_time(predict_std)

    return {
        "model_name": model_name,
        "gpu_model": gpu_model,
        "optimize": optimize,
        "fit_mean": fit_mean,
        "fit_std": fit_std,
        "predict_mean": predict_mean,
        "predict_std": predict_std,
        "optimize_formatted": optimize_formatted,
        "fit_mean_formatted": fit_mean_formatted,
        "fit_std_formatted": fit_std_formatted,
        "predict_mean_formatted": predict_mean_formatted,
        "predict_std_formatted": predict_std_formatted,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory with extracted time duration csv files",
        nargs="+",
    )

    args = parser.parse_args()
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

    # Store all model data by directory
    model_data_by_dir = {dir_name: [] for dir_name in args.data_dir}

    for data_dir in args.data_dir:
        data_dir_path = Path(data_dir)
        if not data_dir_path.is_dir():
            continue

        for file in data_dir_path.glob("*.csv"):
            # Extract model data
            model = extract_model_data(file)
            # Append data with an extra identifier for the dataset (directory stem)
            model["dataset"] = data_dir_path.stem[
                0
            ]  # First letter of the directory name
            model["ds_label"] = labels.get(model["dataset"], model["dataset"])
            model_data_by_dir[data_dir].append(model)

            # Add events per second
            eps = ds_lengths[model["dataset"]] / model["predict_mean"]
            model["eps"] = eps

    # Process each model and print the corresponding data in LaTeX format
    models = set()
    latex_rows = []
    for dir_name, model_list in model_data_by_dir.items():
        for model in model_list:
            models.add(model["model_name"])

    # Sort models alphabetically or by any custom order
    data_by_model = {model: {} for model in models}
    for dir_name, model_list in model_data_by_dir.items():
        for model in model_list:
            if model["model_name"] not in data_by_model:
                data_by_model[model["model_name"]] = {}
            data_by_model[model["model_name"]][dir_name] = model

    # Sort models by the sum of their fit means across all datasets
    models = sorted(
        models,
        key=lambda m: sum(
            data_by_model[m].get(dir_name, {"predict_mean": []})["predict_mean"]
            for dir_name in args.data_dir
        ),
    )

    for model_name in models:
        # Create a list of all the data for the current model across all datasets
        sorted_data = []
        for data_dir in args.data_dir:
            sorted_data.extend(
                [
                    m
                    for m in model_data_by_dir[data_dir]
                    if m["model_name"] == model_name
                ]
            )

        # Sort the data by Fit Mean (training time)
        order = {"B": 0, "H": 1, "T": 2}
        sorted_data = sorted(sorted_data, key=lambda x: order[x["dataset"]])

        # Prepare rows for the LaTeX table
        for d in sorted_data:
            row = {
                "Model": model_name,
                "Dataset": d["ds_label"],
                "GPU": d["gpu_model"] if d["gpu_model"] else "",
                "Fit Mean": d["fit_mean_formatted"],
                "Fit Std": d["fit_std_formatted"],
                "Predict Mean": d["predict_mean_formatted"],
                "Predict Std": d["predict_std_formatted"],
                "KEPS": f"{d['eps'] / 1000:.1f}",
            }
            latex_rows.append(row)

    # Convert rows to a DataFrame
    latex_df = pd.DataFrame(latex_rows).set_index(["Model", "Dataset"])
    print(latex_df)

    # Generate LaTeX table
    latex_table = latex_df.to_latex(
        index=True,
    )
    latex_table = latex_table.replace("\\multirow[t]", "\\multirow[c]")
    latex_table = latex_table.replace("\\cline{1-", "\\cline{2-")

    print(latex_table)
