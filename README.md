# Comparative study of anomaly-detection methods in logs

This repository integrates multiple open-source implementations of anomaly detection methods for log data. The goal is to provide a unified comparison evaluating these methods on different datasets. The repository includes the following methods:


| Source Repository                                                           | Method Name | Description                                                  |
|-----------------------------------------------------------------------------|-------------|--------------------------------------------------------------|
| [`logpai/loglizer`](https://github.com/logpai/loglizer)                     | SVM         | Support Vector Machine                                       |
| └─                                                                          | LogCluster  | Clustering-based log anomaly detection                       |
| [`LeonYang95/SemPCA`](https://github.com/LeonYang95/SemPCA)                 | SemPCA      | PCA combined with GloVe embeddings                           |
| ├─                                                                          | PCA         | Principal Component Analysis                                 |
| ├─                                                                          | DeepLog     | LSTM predicting next log template                            |
| ├─                                                                          | LogAnomaly  | LSTM predicting next log template based on GloVe Embeddings  |
| └─                                                                          | LogRobust   | LSTM classifying anomalies based on GloVe Embeddings         |
| [`HelenGuohx/logbert`](https://github.com/HelenGuohx/logbert)               | LogBERT     | Log Anomaly Detection via BERT                               |
| [`LogIntelligence/NeuralLog`](https://github.com/LogIntelligence/NeuralLog) | NeuralLog   | A transformer-based classification model without log parsing |

## Requirements

Tested on **`python3.9`**, and it is **strongly recommended** to use this version. As this repository integrates multiple submodules that are forks of other experimental repositories and paper artifacts, there are a lot of dependency conflicts which were resolved specifically for this version of Python. Please use a virtual environment of your choice to avoid conflicts with your system Python installation.

The build requirements outside those listed in the `requirements.txt` are:

```shell
pip                           25.0.1
setuptools                    53.0.0
wheel                         0.45.1
```

With these packages installed (see if `pip list | grep "setuptools\|wheel\|pip"` matches the output above) the following packages can be installed with:

```shell
pip install -r requirements.txt
```

## Setup

Before first usage, please examine the configuration file `paths.toml` and adjust the paths to your local setup. The default paths are set to the following:

```toml
datasets = "datasets"
outputs = "outputs"
artefacts = "artefacts"
cache = "cache"
```

The `datasets` directory is where the datasets will be downloaded and stored along with parsing artefacts and cache files of different data representations by the dataloaders.

The `outputs` directory is where the results of the experiments will be stored in a directory structure `outputs/${DATASET_ID}/${METHOD_ID}/`. There you will find the outputs of the methods. If output metrics of a method are already present in the output directory, the method will not be run again.

The `artefacts` directory is where the artefacts of the methods will be stored. Every run of the anomaly detection pipeline will create a new directory in the `artefacts` directory and store the artefacts there. This is useful to examine training artefacts of some of the methods or compare outputs of different runs of the same method.

The `cache` directory is where the cache files of the methods are stored, as some methods require saving models or other files on disk to function. If you want to clear the cache, you can delete the `cache` directory anytime.

**Important**: Create a symbolic link from your selected `datasets` directory to the `datasets` directory within the sempca.d submodule, as the code there sadly still relies on a few hardcoded paths and **will not work without this**.

```shell
ln -s /path/to/your/datasets /path/to/repo/logadcomp/sempca.d/datasets
```

## Running the code

To run the pipeline, use the `orchestrator.py` script. Here is an example usage:

```shell
python orchestrator.py HDFS SemPCA --shuffle --train_ratio 0.1
```

This will run the `SemPCA` method on the `HDFS` dataset with a training ratio of 0.1 and shuffle the data before splitting it into training and test sets. The results will be stored in the `outputs` directory. The orchestrator will run 10 cross-validation runs with these settings.

The first run of the orchestrator script will include the download and parsing of the selected dataset. This will take some time, but is done only once for the dataset, all subsequent runs will use the cached results of the parsing. 

If you want to run only some of the splits, you can use the `--splits` argument. For example, to run only the first 3 splits, use:

```shell
python orchestrator.py HDFS SemPCA --shuffle --train_ratio 0.1 --splits "0-2"  # or --splits "0,1,2"
```

To see all available options, use:

```shell
python orchestrator.py --help
```

The repository also contains all necessary scripts to run experiments on a PBS scheduled HPC cluster like the [MetaCentrum computing grid service](https://docs.metacentrum.cz/en/docs/welcome). The scripts are located in the `pbs` directory, and you can learn more about it [here](pbs/README.md).

## Visualize outputs

Once the pipeline is finished, you can visualize the outputs of the methods using the `plot.py` script. This script will create plots of the results within the `outputs` directory and saves them alongside the results. The script expects the directories with outputs for a specific dataset as input. For example, to visualize the outputs of the `HDFS` dataset from the example above, use:

```shell
python plot.py outputs/HDFSShuffled_0.1
```

This will generate plots for all methods evaluated with these settings. 

There are also additional scripts that produce the figures used in the masters thesis. These scripts are located in the `plots` directory and are further described [here](plots/README.md).

## Datasets

The datasets are not included in this repository due to their size. However, the prepared dataloaders will download the datasets automatically when running the code. The datasets are:

- HDFS
- BGL
- Thunderbird