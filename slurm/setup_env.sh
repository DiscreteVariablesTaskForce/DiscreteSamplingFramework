#!/bin/bash -l
# One-off: build the virtualenv the jobs run in. Run it on a login node, from
# anywhere in the clone: pip needs the network, which compute nodes lack.
#
#   export DSF_MODULE="python/3.11.9-gcc14.2.0 python-venv/1.0-gcc14.2.0"
#   export DSF_VENV=$HOME/venvs/dsf     # anywhere the compute nodes can read
#   bash slurm/setup_env.sh
#
# Put the two exports in your ~/.bashrc as well: the jobs are submitted with
# --export=ALL and load the same module and virtualenv on the compute node.
#
# mpi4py and smccomponents are only needed for the MPI executor, which these
# runs don't use, and mpi4py needs an MPI library to build against. They are
# left out unless WITH_MPI=1, in which case add an MPI module to DSF_MODULE.
set -euo pipefail

cd "$(dirname "$0")/.."
: "${DSF_VENV:?set DSF_VENV to the directory the virtualenv should go in}"

module purge
if [ -n "${DSF_MODULE:-}" ]; then
    # shellcheck disable=SC2086  # deliberate word splitting: one load per module
    module load $DSF_MODULE
fi

python3 -m venv "$DSF_VENV"
# shellcheck disable=SC1091
source "$DSF_VENV/bin/activate"
python -m pip install --upgrade pip

if [ "${WITH_MPI:-0}" = 1 ]; then
    python -m pip install -r requirements.txt
else
    requirements=$(mktemp)
    grep -v -E '^(mpi4py|smccomponents)' requirements.txt > "$requirements"
    python -m pip install -r "$requirements"
    rm -f "$requirements"
fi
python -m pip install -e .

python -c "import discretesampling.domain.incremental_decision_tree"
echo "Environment ready in $DSF_VENV. Check the experiment list with:"
echo "  python examples/incremental_decision_tree/run_experiments.py --list"
