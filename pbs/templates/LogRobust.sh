export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET LogRobust --n_trials 30 \
  $OPTS
