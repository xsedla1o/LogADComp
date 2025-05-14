# PBS scripts

This directory contains the PBS scripts used to run the experiments using the [MetaCentrum computing grid service](https://docs.metacentrum.cz/en/docs/welcome).
The scripts can be used to run the experiments there or on any other computing cluster that uses the PBS job scheduler.


## Setup

The scripts are designed to run with a set directory structure (relative from the home directory). 
If modifications are required, please adjust the paths in the `setup.sh`, `teardown.sh` and `queue_pbs.py` scripts accordingly.

```
artefacts/  # directory for the artefacts
outputs/    # directory for the outputs
jobs/       # directory for support scripts, move from scripts to this location
├── scripts/  # directory for the generated PBS scripts
├── setup.sh
└── teardown.sh
logadcomp/  # directory for the logadcomp repository
menv.tar.gz # tarball of the virtual environment, see below
```

## Virtual environment

The scripts are designed to run with a packaged virtual environment that contains all the required dependencies.
To create the virtual environment, we used `mamba` and the `conda-pack` package.

1. Create a new conda environment with the required dependencies:

```shell
mamba create --prefix /storage/city/home/user_name/my_new_env python=3.9
pip install -r requirements.txt
```

2. Package the environment using `conda-pack`:

```shell
mamba install -c conda-forge conda-pack
conda pack --prefix /storage/city/home/user_name/my_new_env -o menv.tar.gz
mamba deactivate
```

3. Move the `menv.tar.gz` file to the home directory.

The `setup.sh` script will now be able to use the environment to run the experiments.
For completeness, the manual usage of the packed environment is as follows:

```shell
# Copy to wherever
cp menv.tar.gz /path/to/dst/

# Unpacking
mkdir /path/to/dst/mamba-env
tar -xzf /path/to/dst/my_env.tar.gz -C /path/to/dst/mamba-env
/path/to/dst/mamba-env/bin/conda-unpack

# Ta da
mamba activate /path/to/dst/mamba-env
```


## Example usage

Individual job scripts can be created and executed in bulk using the `queue_pbs.py` script. 
Check out the script help for the full options. An example of the script usage is like this:

```shell
cd jobs 
python ../logadcomp/pbs/queue_pbs.py HDFSLogHub BGL40 BGL120 \
  -t 0.5 0.1 \
  -m LogBERT LogAnomaly DeepLog LogRobust NeuralLog \
  --shuffle
```

Once the jobs finish, the output logs can be associated to the corresponding job output using the `move_to_artefacts.py` script. 
This script will move the output logs to the `artefacts` directory. The script can be run like this, assuming the jobs were started from the current directory:

```shell
python move_to_artefacts.py * ../artefacts/
```

In case a job fails due to running out of time or other error, the job intermittent results can be moved to the central storage using the `salvage.sh` script. 
The script expects a job ID as an argument and will move all relevant job outputs to the long-term storage, finally cleaning up the job scratch directory. The script can be run like this:

```shell
bash salvage.sh 123456789.pbs-m1.metacentrum.cz --clean
```