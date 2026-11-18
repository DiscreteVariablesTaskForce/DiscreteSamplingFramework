#!/bin/bash -l
# One experiment of an experiment list per array task. Submitted as an array
# by slurm/submit.sh; every argument after the script name is forwarded to
# run_experiments.py, and --index picks this task's experiment out of the list.
#
#   mkdir -p logs   # Slurm opens --output before this script runs
#   sbatch --array=0-13 slurm/experiment.sbatch --run-id my_sweep
#
# The -l on line 1 is required on Barkla: without a login shell `module load`
# is not defined. Submit from scratch or fastscratch: compute nodes cannot
# write to home or data.
#
#SBATCH -J dsf
# Current working directory and current environment (both are the defaults).
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p nodes
# One node: the chains run in one process pool, so cores spread over several
# nodes would leave every node but the first idle.
#SBATCH -N 1
# Cores = chains. Override on the command line, as submit.sh does.
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.out
#
# Memory is deliberately not requested: Barkla sets it per core (~9.1 GB on the
# standard nodes), which covers a chain on susy (~4.2 GB at most, while
# parsing the CSV; far less from the copy submit.sh prepares).

set -euo pipefail

# Site-specific: point DSF_MODULE / DSF_VENV at your Python and virtualenv.
# DSF_MODULE may name several modules, e.g.
#   export DSF_MODULE="python/3.11.9-gcc14.2.0 python-venv/1.0-gcc14.2.0"
module purge
if [ -n "${DSF_MODULE:-}" ]; then
    # shellcheck disable=SC2086  # deliberate word splitting: one load per module
    module load $DSF_MODULE
fi
if [ -n "${DSF_VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$DSF_VENV/bin/activate"
fi

# One BLAS thread per worker process: each of the allocated cores would
# otherwise start its own thread pool and they would fight over the allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Datasets/ and Results/ are relative paths.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=== $(date) | task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) | ${SLURM_NTASKS:-?} cores ==="

# --skip-existing: resubmitting the same run id reruns only what is missing.
python examples/incremental_decision_tree/run_experiments.py \
    --index "${SLURM_ARRAY_TASK_ID:-0}" --skip-existing --jobs "${SLURM_NTASKS:-1}" "$@"

echo "=== $(date) | task ${SLURM_ARRAY_TASK_ID:-0} finished ==="
