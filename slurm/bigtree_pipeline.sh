#!/bin/bash
# The large-tree runs in one go (see examples/incremental_decision_tree/experiments_bigtree.py):
#
#   1. the pilot: SUSY and HEPMASS, MH / DA / HINTS, a few chains, the full budget,
#      a safety cap of 20,000 nodes, no evaluation;
#   2. a 1-core job, held on the pilot, that sets the cap from the largest tree the
#      pilot reached (decide_tree_cap.py) and writes Results/bigtree_pilot/tree_cap.json;
#   3. the four main runs (MCMC and SMC, with and without HEPMASS), submitted now but
#      held on that job, so they start with the cap it chose.
#
# Run from the clone, with the virtualenv active:
#
#   module load $DSF_MODULE; source $DSF_VENV/bin/activate
#   slurm/bigtree_pipeline.sh
#
# Settings from the environment: PILOT_TIME (24:00:00), MAIN_TIME (24:00:00),
# SMC_TIME (24:00:00), BIG_RESULTS_DIR (e.g. /mnt/fastscratch2/users/$USER/dsf_results:
# each run's Results/<id> becomes a link into it, for the space), DRY_RUN=1 (print only).
set -euo pipefail
cd "$(dirname "$0")/.."

E=examples/incremental_decision_tree/experiments_bigtree.py
PILOT_ID=bigtree_pilot
PILOT_TIME="${PILOT_TIME:-24:00:00}"
MAIN_TIME="${MAIN_TIME:-24:00:00}"
SMC_TIME="${SMC_TIME:-24:00:00}"
DRY_RUN="${DRY_RUN:-0}"

link() {   # Results/<id> -> $BIG_RESULTS_DIR/<id>, when asked for
    if [ -n "${BIG_RESULTS_DIR:-}" ] && [ ! -e "Results/$1" ]; then
        mkdir -p "$BIG_RESULTS_DIR/$1" Results
        ln -s "$BIG_RESULTS_DIR/$1" "Results/$1"
        echo "Results/$1 -> $BIG_RESULTS_DIR/$1"
    fi
}
job_of() { awk -v what="$1" '$1 == ">>>" && $2 == what {print $4}'; }

if [ -e "Results/$PILOT_ID/tree_cap.json" ]; then
    echo "Results/$PILOT_ID/tree_cap.json already exists: remove it to decide the cap again" >&2
    exit 1
fi

# 1. The pilot.
link "$PILOT_ID"
out=$(RUN_ID="$PILOT_ID" EXPERIMENTS="$E::pilot" PREP_CORES=4 CORES=4 TIME="$PILOT_TIME" NO_EVAL=1 \
      DRY_RUN="$DRY_RUN" slurm/submit.sh)
echo "$out"
if [ "$DRY_RUN" = 1 ]; then PILOT_JOB="<pilot>"; else PILOT_JOB=$(echo "$out" | job_of sampling); fi

# 2. The decision, after the pilot (afterany: a chain that timed out still counts).
DECIDE=(sbatch --parsable --dependency="afterany:$PILOT_JOB" slurm/decide.sbatch --pilot-dir "Results/$PILOT_ID")
if [ "$DRY_RUN" = 1 ]; then echo "${DECIDE[*]}"; DECIDE_JOB="<decide>"
else DECIDE_JOB=$("${DECIDE[@]}"); DECIDE_JOB="${DECIDE_JOB%%;*}"; fi
echo ">>> decision  job $DECIDE_JOB   (held on afterany:$PILOT_JOB)"

# 3. The main runs, held on the decision.
submit() {   # run id, group, then KEY=VALUE settings for submit.sh
    local id=$1 group=$2; shift 2
    link "$id"
    env RUN_ID="$id" EXPERIMENTS="$E::$group" AFTER="$DECIDE_JOB" DRY_RUN="$DRY_RUN" "$@" slurm/submit.sh
}
submit ch3_bigtree mcmc SPLIT=4 CORES=16 TIME="$MAIN_TIME" EVAL_CORES=64 EVAL_TIME=24:00:00
submit ch4_smc_bigtree smc CORES=16 TIME="$SMC_TIME" EVAL_CORES=32 EVAL_TIME=24:00:00
submit ch3_bigtree_hepmass mcmc_hepmass PREP_CORES=4 SPLIT=4 CORES=16 TIME="$MAIN_TIME" EVAL_CORES=64 EVAL_TIME=24:00:00
submit ch4_smc_bigtree_hepmass smc_hepmass PREP_CORES=4 CORES=32 TIME="$SMC_TIME" EVAL_CORES=32 EVAL_TIME=24:00:00
