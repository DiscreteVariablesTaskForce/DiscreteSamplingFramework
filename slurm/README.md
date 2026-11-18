# Running the experiments on Barkla (SLURM)

`slurm/submit.sh` runs an experiment list, by default
`examples/incremental_decision_tree/experiments.py`:

- **Sampling:** a job array with one task per experiment. Each task requests
  one node and `CORES` cores, and runs one chain per core.
- **Evaluation:** one job, held until every sampling task has finished.
- **Data:** each dataset is split once at submission, into the run's
  directory, so the jobs neither re-parse the CSV nor need the network.

Results land in `Results/<run-id>/`, exactly as a local
`run_experiments.py` run would write them, so plotting and the notebooks work
on them unchanged.

## One-off setup, on a login node

1. **Clone the repository under scratch or fastscratch.** Jobs can't be
   submitted from home or data, and compute nodes can't write there:
   ```
   git clone -b adding_hints <repo url>
   cd DiscreteSamplingFramework
   ```
2. **Copy the datasets.** `Datasets/` isn't in git. From your own machine:
   ```
   rsync -av Datasets/SUSY.csv <user>@barkla:<clone>/Datasets/
   ```
   wine, digits and covtype come from scikit-learn. covtype is downloaded at
   submission, on the login node.
3. **Choose your Python and virtualenv.** Put these in `~/.bashrc` too, because
   the jobs load the same ones:
   ```
   export DSF_MODULE="python/3.11.9-gcc14.2.0 python-venv/1.0-gcc14.2.0"
   export DSF_VENV=$HOME/venvs/dsf
   ```
4. **Build the environment.** It leaves out `mpi4py`, which these runs don't
   need and which would need an MPI module; see the script for `WITH_MPI=1`.
   ```
   bash slurm/setup_env.sh
   ```

## Running

Load the environment in your login shell, then submit from the clone:

```
module load $DSF_MODULE && source $DSF_VENV/bin/activate

# First time: a few seconds of everything on wine, to check the set-up.
RUN_ID=smoke EXPERIMENTS=slurm/smoke_experiments.py CORES=2 TIME=0:20:00 \
    EVAL_CORES=2 EVAL_TIME=0:20:00 slurm/submit.sh

# The real thing.
RUN_ID=ch3_ch4 MAIL_USER=you@liverpool.ac.uk slurm/submit.sh
squeue -u $USER
```

- **Check before submitting:** `DRY_RUN=1` prints the `sbatch` commands without
  submitting anything.
- **Validation:** the whole list is checked before anything is queued. Every
  entry must be valid, and no two may write the same file.
- **Logs:** one per task, in `logs/`.
- **Failed or timed-out tasks:** submit the same `RUN_ID` again. Experiments
  whose files are already written are skipped, and so are evaluations that
  already have their metrics. `DEPEND_TYPE=afterany` evaluates whatever
  finished, even if some tasks failed.
- **Evaluation on its own:** `slurm/submit_eval.sh <run-id>` (for example,
  after rerunning failed tasks).
- **Fetching the results:** from your own machine. The prepared data is large
  and can be skipped:
  ```
  rsync -av --exclude data/ <user>@barkla:<clone>/Results/ch3_ch4 Results/
  ```

## Settings, all from the environment

| Variable | Default | |
|---|---|---|
| `RUN_ID` | timestamp | the run's directory under `Results/` |
| `EXPERIMENTS` | `examples/incremental_decision_tree/experiments.py` | the experiment list |
| `CORES` | 16 | cores per sampling task: match the experiments' chains |
| `TIME` | 12:00:00 | time limit per sampling task |
| `EVAL_CORES` | 24 | cores for the evaluation job |
| `EVAL_TIME` | 12:00:00 | time limit for the evaluation job |
| `PARTITION` | nodes | |
| `MAX_PARALLEL` | none | most sampling tasks running at once |
| `DEPEND_TYPE` | afterok | `afterany` evaluates even if some tasks failed |
| `MAIL_USER`, `MAIL_TYPE` | off, BEGIN,END,FAIL | email notifications |

**Memory** is never requested: Barkla allocates about 9.1 GB per core.
That covers one chain on susy, which needs about 4.2 GB at most, when it
parses the CSV.
