export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET DeepLog --n_trials 30 \
  $OPTS
