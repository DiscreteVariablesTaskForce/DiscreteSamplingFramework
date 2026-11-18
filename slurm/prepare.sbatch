#!/bin/bash -l
# Split every dataset an experiment list uses once, into
# <results-root>/<run-id>/data/, before its sampling array starts. Submitted by
# slurm/submit.sh, which holds the array on this job; every argument after the
# script name goes to run_experiments.py.
#
# A job rather than a step on the login node, whose limit is 1 core and 4 GB:
# parsing susy's CSV peaks at about 4.2 GB. One core here comes with ~9.1 GB.
#
#SBATCH -J dsf_prepare
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

module purge
if [ -n "${DSF_MODULE:-}" ]; then
    # shellcheck disable=SC2086  # deliberate word splitting: one load per module
    module load $DSF_MODULE
fi
if [ -n "${DSF_VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$DSF_VENV/bin/activate"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=== $(date) | preparing data on $(hostname) ==="
python examples/incremental_decision_tree/run_experiments.py --prepare "$@"
echo "=== $(date) | data prepared ==="
