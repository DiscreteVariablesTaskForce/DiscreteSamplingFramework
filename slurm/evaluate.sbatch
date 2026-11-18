#!/bin/bash -l
# Evaluate every experiment of a finished run into its _metrics.npz, in one job
# whose workers are shared by all of them. Submitted by slurm/submit.sh, held
# until the sampling array has finished, or by slurm/submit_eval.sh. Every
# argument after the script name goes to evaluate_results.py, so --stride,
# --splits and --overwrite work here too.
#
#   mkdir -p logs
#   sbatch -n 24 -t 12:00:00 slurm/evaluate.sbatch --run-id my_sweep
#
#SBATCH -J dsf_eval
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p nodes
#SBATCH -N 1
# Workers are not chains here: every stored tree of every experiment goes to
# one pool. Each worker loads the data itself, so a very high count costs
# memory and start-up for little gain; 24 suits runs up to a few hundred
# thousand iterations.
#SBATCH -n 24
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#
# Memory is not requested: Barkla allocates it per core.

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

# One BLAS thread per worker: the metrics call scikit-learn, which would
# otherwise start a thread pool inside every one of the allocated cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=== $(date) | evaluation on $(hostname) | ${SLURM_NTASKS:-?} cores ==="

# Experiments that already have their _metrics.npz are skipped.
python examples/incremental_decision_tree/evaluate_results.py --jobs "${SLURM_NTASKS:-1}" "$@"

echo "=== $(date) | evaluation finished ==="
