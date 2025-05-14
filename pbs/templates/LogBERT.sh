export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/sbin:$PATH"
python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET LogBERT --n_trials 1 \
  $OPTS
