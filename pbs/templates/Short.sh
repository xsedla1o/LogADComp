python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET SVM --n_trials 100 \
  $OPTS

python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET PCA --n_trials 100 \
  $OPTS

python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET SemPCA --n_trials 100 \
  $OPTS

# sync the data before running the clustering
rsync -avu $SCRATCHDIR/datasets/ $HOMEDIR/datasets/
rsync -avu $SCRATCHDIR/outputs/ $HOMEDIR/outputs/
rsync -avu $SCRATCHDIR/artefacts/ $HOMEDIR/artefacts/

$HOMEDIR/notify.sh python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET LogCluster --n_trials 50 \
  $OPTS
