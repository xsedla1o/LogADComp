module rm cuda/12.6.1-gcc-10.2.1-hplxoqp
module add cuda/11.6.2-gcc-10.2.1-nwpmxyy

export PATH="/sbin:$PATH"
$HOMEDIR/notify.sh python $SCRATCHDIR/logadcomp/orchestrator.py \
  $DATASET NeuralLog --n_trials 1 \
  $OPTS
