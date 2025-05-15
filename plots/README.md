# Plotting scripts

The scripts in this directory were used to generate the figures and tables for the thesis. 
Some of the scripts source data directly from the experiment outputs,
other use data extracted from the experiment logs or other statistics.
All data to reproduce the figures is included in full.

The scripts were designed to run from the root of the repository, so any relative path issues can be avoided by running the scripts from there.

## F1 score boxplots

The `grouped` directory contains the scripts used to generate the boxplots of the F1 scores for the different methods and datasets.
The settings for the various plots are included in the script itself, so no command line arguments are needed.

```shell
python plots/grouped/plot.py
```

## Resource usage

The `resources` directory contains the scripts used to generate the resource usage plots for the different methods and datasets. The data was extracted from the logs captured during the experiments.
The scripts expect the data directories to be passed as an argument, so the command line usage is as follows:

```shell
python plots/resources/show.py plots/resources/BGL40 plots/resources/HDFSLogHub plots/resources/TBird
python plots/resources/show_mem.py plots/resources/*_mem
```

## BGL grouping histograms

The `bgl_grouping` directory contains the scripts that visualize the session durations when different grouping strategies are used.

```shell
python plots/bgl_grouping/show.py plots/bgl_grouping/*
python plots/bgl_grouping/show_aps.py plots/bgl_grouping/*
```

## Event distribution

To investigate the issues caused by training models on sequential samples of the dataset, the `event_distribution` directory contains the scripts that visualizes the distribution of events in the training and test sets.

```shell
python plots/event_distribution/show.py plots/event_distribution/HDFSFixed  
python plots/event_distribution/show_lines.py plots/event_distribution/HDFSFixed  
```

