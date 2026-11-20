# Running the experiments on Barkla (SLURM)

`slurm/submit.sh` runs an experiment list, by default
`examples/incremental_decision_tree/experiments.py`, as three chained jobs:

- **Prepare:** one short job (1 core) splits each dataset once, into
  `Results/<run-id>/data/`. The jobs read that copy instead of the CSV.
- **Sampling:** a job array held on the prepare job, with one task per
  experiment. Each task requests one node and `CORES` cores, and runs one chain
  per core.
- **Evaluation:** one job, held until the sampling has finished.

Results land in `Results/<run-id>/`, exactly as a local
`run_experiments.py` run would write them, so plotting and the notebooks work
on them unchanged.

## Where things go on Barkla

| What | Where | Why |
|---|---|---|
| the clone, `Datasets/`, `Results/`, `logs/` | `/mnt/scratch/users/$USER` (2 TB, 300k files), or `/mnt/fastscratch/users/$USER` (500 GB) if scratch is near its quota | jobs must be submitted from scratch or fastscratch; neither is backed up, and fastscratch has no disk redundancy |
| the virtualenv | `/mnt/fastscratch/users/$USER/venvs/dsf` (500 GB, 500k files) | where Barkla wants Python installs; home has a 100k-file quota |
| finished results to keep | `~/data` (2.5 TB, backed up) | read-only on compute nodes, so copy results there from a login or visualisation node |

## One-off setup

Before first use, Barkla requires you to accept its usage policy and join the
HPC-USERS mailing list (§2.1.8 of the Barkla docs).

1. **Check your quotas, then clone into scratch or fastscratch.** Past the soft
   quota, Lustre allows a week's grace before it stops all writes, which would
   fail jobs partway:
   ```
   /opt/apps/user_tools/display_disk_usage_quotas.sh
   cd /mnt/scratch/users/$USER         # or /mnt/fastscratch/users/$USER
   git clone -b adding_hints <repo url>
   cd DiscreteSamplingFramework
   ```
2. **Copy the datasets.** `Datasets/` isn't in git. From your own machine:
   ```
   rsync -av Datasets/SUSY.csv <user>@barklalogin1.liv.ac.uk:/mnt/scratch/users/<user>/DiscreteSamplingFramework/Datasets/
   ```
   wine and digits ship with scikit-learn. covtype is downloaded on first use,
   and compute nodes may not reach the network, so if you use it, fetch it
   once on a visualisation node:
   `python -c "from sklearn.datasets import fetch_covtype; fetch_covtype()"`
3. **Set your Python and virtualenv.** Put these in `~/.bashrc` too, because
   the jobs load the same ones:
   ```
   export DSF_MODULE="miniforge3/25.3.0-python3.12.10"
   export DSF_VENV=/mnt/fastscratch/users/$USER/venvs/dsf
   ```
   - The Python must be 3.11 or later; Barkla's default `python3` is 3.9.
   - The code is checked on 3.11 and 3.12. The `python/3.11.9-gcc14.2.0
     python-venv/1.0-gcc14.2.0` modules from your other repo work as well.
4. **Build the environment on a visualisation node** (barklaviz1 or barklaviz2),
   because installs belong there, not on a login node:
   ```
   bash slurm/setup_env.sh
   ```
   It leaves out `mpi4py`, which these runs don't need and which would need
   an MPI module. See the script for `WITH_MPI=1`.

## Running

From a login node, in the clone:

```
module load $DSF_MODULE && source $DSF_VENV/bin/activate

# First time: a few seconds of everything on wine, to check the set-up.
RUN_ID=smoke EXPERIMENTS=slurm/smoke_experiments.py CORES=2 TIME=0:20:00 \
    EVAL_CORES=2 EVAL_TIME=0:20:00 slurm/submit.sh

# The real thing.
RUN_ID=ch34 TIME=6:00:00 MAIL_USER=you@liverpool.ac.uk slurm/submit.sh
RUN_ID=reference EXPERIMENTS=examples/incremental_decision_tree/experiments_reference.py \
    TIME=6:00:00 slurm/submit.sh
RUN_ID=hints_2x4 EXPERIMENTS=examples/incremental_decision_tree/experiments_hints_shapes.py \
    TIME=6:00:00 slurm/submit.sh
squeue -u $USER
```

- **Login-node limit:** `submit.sh` only checks and lists the experiments
  there, to stay within the 1 core and 4 GB a login node allows. The data is
  split by the prepare job.
- **Check before submitting:** `DRY_RUN=1` prints the three `sbatch` commands
  without submitting anything.
- **Validation:** the whole list is checked before anything is queued. Every
  entry must be valid, and no two may write the same file.
- **Logs:** one per job and per task, in `logs/`.
- **Failed or timed-out tasks:** submit the same `RUN_ID` again. Experiments
  whose files are already written are skipped, and so are evaluations that
  already have their metrics. `DEPEND_TYPE=afterany` evaluates whatever
  finished, even if some tasks failed.
- **Evaluation on its own:** `slurm/submit_eval.sh <run-id>`.
- **Keeping the results:** copy finished runs to `~/data`, which is backed up,
  from a login node. The prepared data is large and can be left out:
  ```
  rsync -av --exclude data/ /mnt/scratch/users/$USER/DiscreteSamplingFramework/Results/ch34 ~/data/
  ```
- **Fetching them to your own machine:**
  ```
  rsync -av --exclude data/ <user>@barklalogin1.liv.ac.uk:/mnt/scratch/users/<user>/DiscreteSamplingFramework/Results/ch34 Results/
  ```

## Usage, quotas and limits

```
/opt/apps/user_tools/display_disk_usage_quotas.sh       # every quota and how much is used
/opt/apps/user_tools/display_disk_usage_quota_bars.sh   # the same, as bars
/opt/apps/user_tools/partition_usage.sh nodes 30        # CPU-hours on `nodes` by user, last 30 days
squeue -u $USER                                         # your queued and running jobs
```

- **Jobs:** at most 1,000 per user, counting every array task. A full run
  here is 44 sampling tasks plus 3 prepare and 3 evaluation jobs.
- **Cores:** running jobs are capped per user (by QOS). Tasks beyond the cap
  wait their turn; `MAX_PARALLEL` makes the cap explicit.
- **The `nodes` partition:** 168 cores per node, about 9.1 GB per core, and at
  most 3 days per job. Memory is never requested: it comes with the cores.

## Settings, all from the environment

| Variable | Default | |
|---|---|---|
| `RUN_ID` | timestamp | the run's directory under `Results/` |
| `EXPERIMENTS` | `examples/incremental_decision_tree/experiments.py` | the experiment list |
| `CORES` | 16 | cores per sampling task: match the experiments' chains |
| `TIME` | 12:00:00 | time limit per sampling task (at most 3 days on `nodes`) |
| `PREP_CORES` | 1 | cores for the prepare job; memory comes with them (about 9 GB each), and hepmass needs about 12 GB to split, so 2 or more |
| `PREP_TIME` | 0:30:00 | time limit for the prepare job |
| `EVAL_CORES` | 24 | cores for the evaluation job |
| `EVAL_TIME` | 12:00:00 | time limit for the evaluation job |
| `PARTITION` | nodes | |
| `MAX_PARALLEL` | none | most sampling tasks running at once |
| `DEPEND_TYPE` | afterok | `afterany` evaluates even if some tasks failed |
| `MAIL_USER`, `MAIL_TYPE` | off, BEGIN,END,FAIL | email notifications |

Work run on Barkla has to be acknowledged in published outputs, including the
thesis (§2.1.6 and §21 of the Barkla docs).
