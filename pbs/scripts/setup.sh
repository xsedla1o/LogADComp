#!/bin/bash
# Source this script to set up the environment for the job
HOMEDIR="/storage/brno2/home/$USER/"
export TMPDIR=$SCRATCHDIR

# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

#copy file to scratch, preserving metadata
cp -a $HOMEDIR/datasets $HOMEDIR/outputs $HOMEDIR/logadcomp $HOMEDIR/menv.tar.gz $SCRATCHDIR
cd $SCRATCHDIR/logadcomp || { echo >&2 "Cannot change to working directory!"; exit 1; }

# Load modules
module add python/3.9.12-gcc-10.2.1-rg2lpmk
module add cuda/12.6.1-gcc-10.2.1-hplxoqp
module add mambaforge

# Write paths to paths.toml
cat <<EOF > $SCRATCHDIR/logadcomp/paths.toml
datasets = "$SCRATCHDIR/datasets"
outputs = "$SCRATCHDIR/outputs"
EOF

# Unpack the mamba environment tar
mkdir $SCRATCHDIR/mamba-env
tar -xzf $SCRATCHDIR/menv.tar.gz -C $SCRATCHDIR/mamba-env
rm $SCRATCHDIR/menv.tar.gz
$SCRATCHDIR/mamba-env/bin/conda-unpack
mamba activate $SCRATCHDIR/mamba-env

# Reinstall the sempca.d subdirectory to the new path
pip install -e $SCRATCHDIR/logadcomp/sempca.d
# Link the internal sempca dataset path to the new path
ln -s $SCRATCHDIR/datasets $SCRATCHDIR/logadcomp/sempca.d/datasets

# Check if the GPU is available
nvidia-smi
