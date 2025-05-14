#!/bin/bash
# Source this script to tear down the environment after the job
HOMEDIR="/storage/brno2/home/$USER/"
export TMPDIR=$SCRATCHDIR

mamba deactivate

# copy files back
rsync -avu $SCRATCHDIR/datasets/ $HOMEDIR/datasets/
rsync -avu $SCRATCHDIR/outputs/ $HOMEDIR/outputs/
rsync -avu $SCRATCHDIR/logadcomp/sempca.d/logs/ $HOMEDIR/logadcomp/sempca.d/logs/
rsync -avu $SCRATCHDIR/artefacts/ $HOMEDIR/artefacts/

clean_scratch

