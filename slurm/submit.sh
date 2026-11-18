#!/bin/bash
# Submit an experiment list and its evaluation in one go: a job array with one
# task per experiment, and an evaluation job held until the array has finished,
# so both go in now and the metrics are waiting when you come back.
#
#   slurm/submit.sh                                   # experiments.py, id = timestamp
#   RUN_ID=ch3 slurm/submit.sh                        # a named run, or resume one
#   RUN_ID=ref EXPERIMENTS=path/to/list.py slurm/submit.sh
#   RUN_ID=ch3 DRY_RUN=1 slurm/submit.sh              # print the sbatch commands only
#
# Arguments before `--` go to run_experiments.py, arguments after it go to
# evaluate_results.py:
#
#   RUN_ID=ch3 slurm/submit.sh -- --stride 10
#
# Resources, mail and the rest come from the environment:
#
#   RUN_ID=ch3 CORES=10 TIME=12:00:00 EVAL_CORES=24 EVAL_TIME=12:00:00 \
#   MAIL_USER=you@liverpool.ac.uk slurm/submit.sh
#
# CORES is what each sampling task requests (-N 1 -n CORES). Match it to the
# experiments' chains: a task runs one chain per core, so chains beyond CORES
# queue inside the task and cores beyond the chain count sit idle.
#
# DEPEND_TYPE=afterok (the default) runs the evaluation only if every sampling
# task succeeded; afterany runs it regardless, evaluating whatever finished.
# Resubmitting the same RUN_ID reruns only the experiments whose files are
# missing, and evaluates only those without metrics.
#
# Before running this: load your Python and activate the virtualenv (see
# slurm/README.md), and run it from your clone under scratch or fastscratch --
# jobs may not be submitted from home or data, and compute nodes cannot write
# there. Memory is never requested: Barkla allocates it per core.
set -euo pipefail

cd "$(dirname "$0")/.."

# Split the command line at `--`: run_experiments.py args, then evaluate_results.py args.
RUN_ARGS=()
EVAL_ARGS=()
after_separator=0
for arg in "$@"; do
    if [ "$after_separator" -eq 0 ] && [ "$arg" = "--" ]; then
        after_separator=1
        continue
    fi
    if [ "$after_separator" -eq 0 ]; then
        RUN_ARGS+=("$arg")
    else
        EVAL_ARGS+=("$arg")
    fi
done

RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M)}"
EXPERIMENTS="${EXPERIMENTS:-examples/incremental_decision_tree/experiments.py}"
RESULTS_ROOT="${RESULTS_ROOT:-Results}"
PARTITION="${PARTITION:-nodes}"
CORES="${CORES:-16}"
TIME="${TIME:-12:00:00}"
EVAL_CORES="${EVAL_CORES:-24}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
DEPEND_TYPE="${DEPEND_TYPE:-afterok}"
MAX_PARALLEL="${MAX_PARALLEL:-}"          # e.g. 8: at most 8 tasks at once
DRY_RUN="${DRY_RUN:-0}"

MAIL_OPTS=()
if [ -n "${MAIL_USER:-}" ]; then
    MAIL_OPTS=(--mail-user="$MAIL_USER" --mail-type="${MAIL_TYPE:-BEGIN,END,FAIL}")
fi

if ! python -c "import discretesampling" 2>/dev/null; then
    echo "python cannot import discretesampling: load your Python and activate the" >&2
    echo "virtualenv first, e.g. module load \$DSF_MODULE; source \$DSF_VENV/bin/activate" >&2
    exit 1
fi

RUNNER=examples/incremental_decision_tree/run_experiments.py
WHERE=(--file "$EXPERIMENTS" --run-id "$RUN_ID" --results-root "$RESULTS_ROOT")

# 1. Check the whole list (every entry valid, no two writing the same file)
#    before anything is queued, and size the arrays from it.
N=$(python "$RUNNER" --file "$EXPERIMENTS" --count ${RUN_ARGS[@]+"${RUN_ARGS[@]}"})
python "$RUNNER" --file "$EXPERIMENTS" --list ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} | cut -f1,2 | sed 's/^/  /'

# 2. Split every dataset once, here, into the run's directory. The jobs read
#    that copy, so none of them re-parses the CSV or needs the network for a
#    scikit-learn download.
case "$RESULTS_ROOT" in
    /*) export DSF_DATA_CACHE="$RESULTS_ROOT/$RUN_ID/data" ;;
    *)  export DSF_DATA_CACHE="$PWD/$RESULTS_ROOT/$RUN_ID/data" ;;
esac

array="0-$((N - 1))"
if [ -n "$MAX_PARALLEL" ]; then array+="%$MAX_PARALLEL"; fi
SAMPLE_CMD=(sbatch --parsable --array="$array" --job-name="dsf_$RUN_ID"
            --partition="$PARTITION" --nodes=1 --ntasks="$CORES" --time="$TIME"
            ${MAIL_OPTS[@]+"${MAIL_OPTS[@]}"}
            slurm/experiment.sbatch "${WHERE[@]}" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"})
EVAL_CMD=(sbatch --parsable --job-name="dsf_eval_$RUN_ID"
          --partition="$PARTITION" --nodes=1 --ntasks="$EVAL_CORES" --time="$EVAL_TIME"
          ${MAIL_OPTS[@]+"${MAIL_OPTS[@]}"})
EVAL_SCRIPT=(slurm/evaluate.sbatch --run-id "$RUN_ID" --results-root "$RESULTS_ROOT"
             ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"})

if [ "$DRY_RUN" = 1 ]; then
    echo "DSF_DATA_CACHE=$DSF_DATA_CACHE"
    echo "${SAMPLE_CMD[*]}"
    echo "${EVAL_CMD[*]} --dependency=$DEPEND_TYPE:<sampling job> ${EVAL_SCRIPT[*]}"
    exit 0
fi

python "$RUNNER" "${WHERE[@]}" --prepare ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
mkdir -p logs

# 3. Sampling: one task per experiment.
RUN_JOB=$("${SAMPLE_CMD[@]}")
RUN_JOB="${RUN_JOB%%;*}"               # --parsable may append ";cluster"

# 4. Evaluation, held on the sampling array.
EVAL_JOB=$("${EVAL_CMD[@]}" --dependency="$DEPEND_TYPE:$RUN_JOB" "${EVAL_SCRIPT[@]}")
EVAL_JOB="${EVAL_JOB%%;*}"

echo
echo ">>> run id    $RUN_ID  ->  $RESULTS_ROOT/$RUN_ID"
echo ">>> sampling  job $RUN_JOB   ($N tasks, $CORES cores each, $TIME)"
echo ">>> metrics   job $EVAL_JOB   ($EVAL_CORES cores, $EVAL_TIME), held on $DEPEND_TYPE:$RUN_JOB"
echo ">>> logs in logs/, watch with: squeue -u \$USER"
