#!/bin/bash
# Evaluate a run on its own: one job for every experiment in Results/<run-id>.
# For a run whose evaluation failed or timed out, or one submitted with
# afterany that should be re-evaluated once its failed tasks have been rerun.
#
#   slurm/submit_eval.sh ch3                  # every stored state
#   slurm/submit_eval.sh ch3 --stride 10      # every 10th
#
# The first argument is the run id; anything after it goes to
# evaluate_results.py. Resources and mail come from the environment:
#
#   EVAL_CORES=24 EVAL_TIME=12:00:00 PARTITION=nodes MAIL_USER=you@liverpool.ac.uk \
#       slurm/submit_eval.sh ch3
#
# DEPEND=afterok:<job id> holds the job until that one has finished.
# Experiments that already have their metrics are skipped unless --overwrite.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
    echo "usage: slurm/submit_eval.sh <run-id> [evaluate_results.py options]" >&2
    exit 1
fi
RUN_ID="$1"
shift

RESULTS_ROOT="${RESULTS_ROOT:-Results}"
PARTITION="${PARTITION:-nodes}"
EVAL_CORES="${EVAL_CORES:-24}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
DEPEND="${DEPEND:-}"

DEPEND_OPTS=()
if [ -n "$DEPEND" ]; then
    DEPEND_OPTS=(--dependency="$DEPEND")
fi
MAIL_OPTS=()
if [ -n "${MAIL_USER:-}" ]; then
    MAIL_OPTS=(--mail-user="$MAIL_USER" --mail-type="${MAIL_TYPE:-BEGIN,END,FAIL}")
fi

# The split submit.sh prepared for this run, if it is there.
case "$RESULTS_ROOT" in
    /*) data="$RESULTS_ROOT/$RUN_ID/data" ;;
    *)  data="$PWD/$RESULTS_ROOT/$RUN_ID/data" ;;
esac
if [ -d "$data" ]; then
    export DSF_DATA_CACHE="$data"
fi

mkdir -p logs
sbatch --job-name="dsf_eval_$RUN_ID" \
       --partition="$PARTITION" \
       --nodes=1 \
       --ntasks="$EVAL_CORES" \
       --time="$EVAL_TIME" \
       ${DEPEND_OPTS[@]+"${DEPEND_OPTS[@]}"} \
       ${MAIL_OPTS[@]+"${MAIL_OPTS[@]}"} \
       slurm/evaluate.sbatch --run-id "$RUN_ID" --results-root "$RESULTS_ROOT" "$@"
