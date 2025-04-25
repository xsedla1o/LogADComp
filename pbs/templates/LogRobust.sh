export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$HOMEDIR/notify.sh python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET LogRobust --n_trials 30 \
  $OPTS
