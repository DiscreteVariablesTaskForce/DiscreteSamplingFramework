#!/bin/bash -l
# Choose the tree-size cap of the large-tree runs from their pilot
# (examples/incremental_decision_tree/decide_tree_cap.py). Submitted by
# slurm/bigtree_pipeline.sh, held on the pilot's sampling array; the main runs
# are held on this job, so they start only once it has written the cap.
#
#SBATCH -J dsf_decide_cap
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
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

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "=== $(date) | deciding the tree cap on $(hostname) ==="
python examples/incremental_decision_tree/decide_tree_cap.py "$@"
echo "=== $(date) | done ==="
