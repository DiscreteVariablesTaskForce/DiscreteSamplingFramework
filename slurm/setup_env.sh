#!/bin/bash -l
# One-off: build the virtualenv the jobs run in. Run it on a visualisation node
# (barklaviz1 or barklaviz2), from anywhere in the clone: pip needs the
# network, and Barkla asks for installs and builds to run there rather than on
# the login nodes.
#
#   export DSF_MODULE="miniforge3/25.3.0-python3.12.10"
#   export DSF_VENV=/mnt/fastscratch/users/$USER/venvs/dsf
#   bash slurm/setup_env.sh
#
# The virtualenv goes on fastscratch, which Barkla recommends for Python
# installs: home has a 100k-file quota and asks for none there. fastscratch has
# no redundancy, so if it is ever lost, rerun this script to rebuild it.
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

# Barkla's own python3 is 3.9; the code is checked on 3.11 and later.
python3 - <<'EOF'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"python3 is {sys.version.split()[0]}: load a module with Python 3.11 "
             "or later in DSF_MODULE first, e.g. miniforge3/25.3.0-python3.12.10")
EOF
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
