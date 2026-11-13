"""
Turn the states a sweep stored into predictive metrics.

sampler_diagnostics.py samples and stores trees; this reads them back and
evaluates them, writing one <experiment>_metrics.npz beside each <experiment>.h5:

    python examples/incremental_decision_tree/sampler_diagnostics.py --dataset wine --run-id wine_baseline
    python examples/incremental_decision_tree/evaluate_results.py --run-id wine_baseline
    python examples/incremental_decision_tree/plot_diagnostics.py --run-id wine_baseline

    python examples/incremental_decision_tree/evaluate_results.py --run-id covtype --name covtype_12p5 \
        --stride 10 --splits test --jobs 16

Why this is a separate step. One evaluation routes every row of a split through
a tree; on covtype that costs an order of magnitude more than the sampler
iteration that produced the tree. Doing it inline made the diagnostics the
dominant cost of a run and fixed, at sampling time, which metrics on which
splits at which thinning you would ever be able to look at. Doing it here means
the run is sampled once and can be evaluated as often and as many ways as you
like -- on the test split now and the training split later, densely over the
first thousand iterations and coarsely over the rest -- off the same file.

It is also much cheaper than the inline version was, for reasons that are all
exact rather than approximations:

  * Each distinct tree is evaluated once per run. A record's metrics are the
    weighted mean of its trees' own metrics, and a tree's metrics don't depend
    on its weight, so an MCMC chain sitting on a rejected state, a resampled
    SMC step holding one tree in many slots, and an SMC particle surviving
    from step to step all cost nothing after the first time.
  * The train split is never routed. A tree's stored leaf counts are exactly
    the training rows in each leaf, and every row in a leaf gets the same
    prediction, so each train metric is a sum over leaves.
  * Only the splits asked for are evaluated, and only every --stride-th record.

What is left is one independent evaluation per distinct tree, so that is what
is spread over the cores: every run of every experiment is cut into chunks of
trees and all of them go to one pool of --jobs workers. A single experiment
with a single long run uses every core just as a whole directory does.

The .npz holds one (n_runs, n_points) array per metric -- the runs are the
chains (MCMC) or the repeats (SMC) -- plus `iterations`, the sampler iteration
or step behind each point, and the confusion matrices as (n_runs, n_points,
K, K). plot_diagnostics.py reads exactly this.
"""
import argparse
import multiprocessing
import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discretesampling.domain import incremental_decision_tree as idt  # noqa: E402
from results_io import (experiment_names, find_experiment_files,  # noqa: E402
                        iter_experiment_hdf5, latest_run_id, load_run_hdf5,
                        metrics_filename, parse_experiment, read_run_config,
                        states_of)
from sampler_diagnostics import DEFAULT_CFG, load_data  # noqa: E402

SPLITS = ("train", "test")

# Bookkeeping written alongside the metrics, but not metrics themselves: they
# describe the columns rather than measuring anything, so `for metric in npz`
# style reads have to be able to tell them apart.
BOOKKEEPING = ("iterations", "stride", "n_runs")


# --------------------------------------------------------------------------- #
# evaluating trees
# --------------------------------------------------------------------------- #

# Loaded once per process rather than once per chunk: a worker evaluates many
# chunks against the same dataset, and re-splitting covtype for each of them
# costs more than the evaluation.
_DATA = {}

# Trees per task. Small enough that one run's trees spread over every worker
# and a bar moves often, large enough that handing a chunk to a worker and its
# metrics back costs nothing beside evaluating it.
CHUNK = 32

# The run this process last evaluated trees from. Only one is kept: a run's
# chunks are queued together, so a worker stays on one run for a while, and
# keeping every run it has seen would cost a copy of the file per worker.
_SERIES = {}


def splits_for(dataset, wanted):
    """[(name, X, y), ...] for the named splits of a dataset."""
    if dataset not in _DATA:
        X_train, X_test, y_train, y_test = load_data(dataset)
        _DATA[dataset] = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return [(name,) + _DATA[dataset][name] for name in wanted]


def tree_metrics(state, splits):
    """
    One tree's metrics on each requested split. Train is read off the tree's
    stored leaf counts, which are exactly the training rows in each leaf
    (check_fitted_to makes sure of that), so only test is ever routed.
    """
    out = {}
    for name, X, y in splits:
        if name == "train":
            out.update(idt.fitted_metrics(state, prefix="train_"))
        else:
            out.update(idt.state_metrics(state, X, y, prefix=f"{name}_"))
    return out


def check_fitted_to(path, series, X, y):
    """
    Refuse to evaluate a run unless its trees were fitted to exactly the train
    split load_data returns now. If the split has changed since sampling, train
    metrics read off the leaf counts are wrong, and so are test metrics: the
    test rows may no longer be held out, or the features may have moved.
    Routing the train split through the run's largest tree and re-tallying
    settles it.
    """
    state = series.state(int(np.argmax(series.node_counts)))
    if not np.array_equal(idt.leaf_tallies(state, X, y), state.counts.block):
        raise SystemExit(
            f"{path}: the stored trees' leaf counts don't match the train split "
            f"load_data gives now, so the data has changed since this run was "
            f"sampled and neither split would be evaluated on the right rows. "
            f"Restore the split the run was sampled on.")


def picked_ensembles(series, stride):
    """
    (slots, weights) for every stride-th record, without the slots that carry
    no weight: they add nothing to the mean, so aren't worth evaluating.
    """
    out = []
    for i in range(0, len(series), stride):
        slots, weights = series.ensemble(i)
        keep = weights > 0
        out.append((slots[keep], weights[keep]))
    return out


def evaluate_chunk(path, run, slots, dataset, wanted_splits):
    """
    The metrics of the stored trees `slots` of Run_<run> of one file, in that
    order. This is the unit of work a worker is handed: it names the trees
    rather than carrying them, and the worker reads the run's states itself.
    """
    if (path, run) not in _SERIES:
        _SERIES.clear()
        _SERIES[path, run] = states_of(load_run_hdf5(path, run, prefix="state_"))
    series = _SERIES[path, run]
    splits = splits_for(dataset, wanted_splits)
    return [tree_metrics(series.state(s), splits) for s in slots]


def record_means(ensembles, per_tree):
    """
    {metric: list over records} for one run, given the (slots, weights) of its
    records, from picked_ensembles, and {slot: metrics} for its trees.

    A record's metrics are the weighted mean of its trees' own metrics, and a
    tree's metrics don't depend on the weight it carries, so each distinct
    stored tree is evaluated once and reused by every record holding it: an
    MCMC chain sitting on a rejected state and an SMC particle surviving from
    step to step cost nothing after the first time.
    """
    cols = defaultdict(list)
    for slots, weights in ensembles:
        mean = idt.weighted_mean([per_tree[s] for s in slots], weights)
        for k, v in mean.items():
            cols[k].append(v)
    return dict(cols)


class FileEvaluation:
    """
    One (experiment file -> .npz) job in progress, in the main process.

    Opening it works out which trees need evaluating; chunks() cuts them into
    evaluate_chunk tasks, add() takes each one's result back in whatever order
    they finish, and save() writes the .npz once `outstanding` reaches zero.
    `bar` is reset to the number of trees to evaluate and ticks as chunks land,
    so only this process ever draws.

    Each run is averaged into its records and checkpointed to
    <experiment>_metrics.partial/ as soon as its last chunk lands, so a long
    file that dies part way -- a crash, a freeze, Ctrl-C -- loses at most the
    runs in flight. Opening the file again picks up every checkpointed run
    that was made from the same .h5 at the same stride and splits, and only
    evaluates the rest. The checkpoints go once the .npz is written.
    """

    def __init__(self, job, bar):
        self.path, self.out_path, self.dataset, self.splits, self.stride = job
        self.bar = bar
        self.started = time.perf_counter()
        self.partial_dir = self.out_path[:-len(".npz")] + ".partial"
        # What a checkpoint has to match to be reused: the same sampled file,
        # thinned the same way, evaluated on the same splits.
        source = os.stat(self.path)
        self.stamp = {"stride": self.stride, "splits": "+".join(self.splits),
                      "source_mtime_ns": source.st_mtime_ns,
                      "source_size": source.st_size}

        # A run at a time: only its states are read, and they are dropped as
        # soon as its ensembles are picked, since the workers read the states
        # they need themselves and this process only averages what they find.
        self.ensembles, self.todo, self.per_tree, self.cols = [], [], [], []
        self.iterations = np.zeros(0, dtype=np.int64)
        for run, arrays in enumerate(iter_experiment_hdf5(self.path, "state_")):
            series = states_of(arrays)
            if series is None:
                raise SystemExit(
                    f"{self.path} holds no states: it was sampled with "
                    f"--no-states, and there is nothing to evaluate. Re-run "
                    f"the sampling without it.")
            if run == 0:
                [(_, X_fit, y_fit)] = splits_for(self.dataset, ("train",))
                check_fitted_to(self.path, series, X_fit, y_fit)
                self.iterations = series.record_iterations()[::self.stride].copy()
            cols = self._checkpointed(run)
            picked = None if cols is not None else picked_ensembles(series, self.stride)
            self.cols.append(cols)
            self.ensembles.append(picked)
            # The distinct trees the run's records hold, which is all that
            # gets evaluated.
            self.todo.append([] if picked is None else
                             sorted({int(s) for slots, _ in picked for s in slots}))
            self.per_tree.append({})

        resumed = sum(c is not None for c in self.cols)
        if resumed:
            tqdm.write(f"    {os.path.basename(self.path)}: {resumed} of "
                       f"{len(self.cols)} runs already evaluated, picking up "
                       f"from {os.path.basename(self.partial_dir)}/")
        self.evaluated = 0
        self.outstanding = 0
        self.left_in_run = [0] * len(self.todo)
        bar.reset(total=sum(len(slots) for slots in self.todo))

    def _checkpoint_path(self, run):
        return os.path.join(self.partial_dir, f"Run_{run}.npz")

    def _checkpointed(self, run):
        """Run `run`'s {metric: per-record array} from a checkpoint, if there is
        one this evaluation can reuse; None if it has to be evaluated."""
        path = self._checkpoint_path(run)
        if not os.path.exists(path):
            return None
        try:
            with np.load(path) as saved:
                if any(saved[f"_{k}"].item() != v for k, v in self.stamp.items()):
                    return None
                return {k: saved[k] for k in saved.files if not k.startswith("_")}
        except BROKEN_NPZ:
            # Cut off by whatever stopped the last attempt.
            return None

    def chunks(self):
        """The evaluate_chunk argument tuples covering every tree to evaluate."""
        out = [(self.path, run, slots[a:a + CHUNK], self.dataset, self.splits)
               for run, slots in enumerate(self.todo)
               for a in range(0, len(slots), CHUNK)]
        for _, run, _, _, _ in out:
            self.left_in_run[run] += 1
        self.outstanding = len(out)
        return out

    def add(self, chunk, metrics):
        """Take one chunk's metrics, as evaluate_chunk returned them."""
        _, run, slots, _, _ = chunk
        self.per_tree[run].update(zip(slots, metrics))
        self.evaluated += len(slots)
        self.bar.update(len(slots))
        self.outstanding -= 1
        self.left_in_run[run] -= 1
        if not self.left_in_run[run]:
            self._finish_run(run)

    def _finish_run(self, run):
        """Average a run whose trees are all in, checkpoint it, and let its
        per-tree metrics go."""
        cols = {k: np.asarray(v) for k, v in
                record_means(self.ensembles[run], self.per_tree[run]).items()}
        os.makedirs(self.partial_dir, exist_ok=True)
        save_npz(self._checkpoint_path(run),
                 {**cols, **{f"_{k}": v for k, v in self.stamp.items()}})
        self.cols[run] = cols
        self.per_tree[run] = self.ensembles[run] = None

    def save(self):
        """Stack the runs' records, write the .npz and report it."""
        per_run = [c if c is not None else {} for c in self.cols]

        # Runs of one experiment share iters/steps and store_every, so they
        # line up; a run cut short (a crash mid-sweep) is truncated to the
        # common length rather than refused, so the rest of the sweep still
        # plots.
        lengths = {len(next(iter(c.values()))) for c in per_run if c}
        n_points = min(lengths) if lengths else 0
        if len(lengths) > 1:
            tqdm.write(f"    [warning] {os.path.basename(self.path)}: runs hold "
                       f"{sorted(lengths)} evaluated records; truncating to "
                       f"{n_points}")

        keys = sorted(set().union(*(c.keys() for c in per_run))) if per_run else []
        matrices = {k: np.stack([np.asarray(c[k][:n_points]) for c in per_run])
                    for k in keys}
        # On disk before the checkpoints it replaces are deleted.
        save_npz(self.out_path,
                 dict(iterations=self.iterations[:n_points], stride=self.stride,
                      n_runs=len(next(iter(matrices.values()))) if matrices else 0,
                      **matrices),
                 compressed=True)
        shutil.rmtree(self.partial_dir, ignore_errors=True)
        finish(self.bar)
        summarise(self.out_path, matrices, self.evaluated,
                  time.perf_counter() - self.started)


# What np.load raises on a .npz cut off mid-write, or left empty by a crash.
BROKEN_NPZ = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


def save_npz(path, arrays, compressed=False):
    """
    np.savez(_compressed) that a crash cannot leave half-written: written
    aside, flushed to disk, then renamed into place, so a file at `path` is
    always a whole one. Without the flush a crash can keep the rename but
    lose the data, leaving an empty file under the final name.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    folder = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(folder)
    finally:
        os.close(folder)


def npz_readable(path):
    """Whether every array of the .npz at `path` loads."""
    try:
        with np.load(path) as saved:
            for key in saved.files:
                saved[key]
        return True
    except BROKEN_NPZ:
        return False


def worker_context():
    """
    How the workers are started: forked on Linux, so they share the datasets
    this process has already loaded instead of each loading a copy -- on susy
    that is a gigabyte and ten seconds per worker. Elsewhere it is the
    platform's default, and each worker loads the data on its first chunk.
    """
    return multiprocessing.get_context(
        "fork" if sys.platform.startswith("linux") else None)


def finish(bar):
    """
    Stop a finished job's bar clock. Bars are only closed once every job is
    done, so without this a bar's elapsed time would keep running to the end
    and its final rate would be diluted by however long the slowest job took.
    """
    bar._time = lambda t=bar._time(): t
    bar.refresh()


# --------------------------------------------------------------------------- #
# driving a whole results directory
# --------------------------------------------------------------------------- #

def plan(results_dir, names, wanted_splits, stride, overwrite):
    """The (path, out_path, dataset, splits, stride) jobs for a run directory."""
    known = experiment_names(read_run_config(results_dir), DEFAULT_CFG)
    if names:
        unknown = [n for n in names if n not in known]
        if unknown and known:
            raise SystemExit(f"no experiment named {', '.join(unknown)} in "
                             f"{results_dir} (it has: {', '.join(known)})")
    else:
        names = known

    # Keyed by the file, since the full-data proposals' files belong to every
    # experiment of their dataset and would otherwise be evaluated once each.
    jobs, skipped = {}, set()
    for name in names:
        try:
            dataset, ss_prop, init = parse_experiment(name)
        except ValueError as err:
            raise SystemExit(str(err)) from None
        for (sampler, proposal), path in find_experiment_files(results_dir, name).items():
            out_path = os.path.join(results_dir, metrics_filename(
                dataset, sampler, proposal, ss_prop, init))
            if out_path in jobs or out_path in skipped:
                continue
            if os.path.exists(out_path) and not overwrite:
                if npz_readable(out_path):
                    print(f">>> Skipping {os.path.basename(path)}: "
                          f"{os.path.basename(out_path)} exists (--overwrite to redo it)")
                    skipped.add(out_path)
                    continue
                print(f">>> {os.path.basename(out_path)} exists but cannot be read "
                      f"(cut off by a crash?): evaluating {os.path.basename(path)} again")
            jobs[out_path] = (path, out_path, dataset, wanted_splits, stride)
    return list(jobs.values())


def summarise(out_path, matrices, evaluated, seconds):
    """One line per finished experiment, ending on what it actually found."""
    line = (f"[{os.path.basename(out_path)}] {evaluated} distinct trees "
            f"evaluated in {seconds:.1f}s")
    for split in SPLITS:
        key = f"{split}_accuracy"
        if key in matrices:
            final = matrices[key][:, -1]
            line += (f"   {split} accuracy {final.mean():.4f}"
                     f" (+/- {final.std():.4f})")
    # tqdm.write, not print, so the line lands above the bars instead of
    # through one of them.
    tqdm.write(line)


def evaluate_in_pool(pool, workers, jobs, bars):
    """
    Evaluate every job on one pool, a chunk of trees per task, so the workers
    stay busy whether the trees are spread over many files or sit in one.
    """
    waiting = iter(zip(jobs, bars))
    owner = {}          # future -> (its FileEvaluation, the chunk, retried yet)
    more = True
    while more or owner:
        # A file is only opened once the workers are about to run short:
        # opening one holds every record's ensemble in memory, and a directory
        # of them opened up front would be held for the whole evaluation.
        while more and len(owner) < 2 * workers:
            job_and_bar = next(waiting, None)
            if job_and_bar is None:
                more = False
                break
            evaluation = FileEvaluation(*job_and_bar)
            chunks = evaluation.chunks()
            if not chunks:
                evaluation.save()
            for chunk in chunks:
                owner[pool.submit(evaluate_chunk, *chunk)] = (evaluation, chunk, False)
        done, _ = wait(owner, return_when=FIRST_COMPLETED)
        for future in done:
            evaluation, chunk, retried = owner.pop(future)
            error = future.exception()
            if error is not None:
                path, run, slots = chunk[:3]
                where = (f"{os.path.basename(path)} Run_{run}, stored trees "
                         f"{slots[0]}..{slots[-1]}")
                if retried:
                    raise RuntimeError(f"{where} failed twice") from error
                # Evaluating a tree is deterministic, so a chunk that fails
                # once and then succeeds was hit by something outside the
                # code -- bad memory, typically. One retry keeps hours of a
                # long run from being thrown away over it, and the warning
                # keeps it from passing unnoticed.
                tqdm.write(f"    [warning] {where}: {error!r}; retrying once. "
                           f"If it then succeeds, suspect the machine (check "
                           f"`journalctl -k` for memory errors), not the data.")
                owner[pool.submit(evaluate_chunk, *chunk)] = (evaluation, chunk, True)
                continue
            evaluation.add(chunk, future.result())
            if not evaluation.outstanding:
                evaluation.save()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="Results")
    p.add_argument("--run-id", default=None,
                   help="results directory under --results-root "
                        "(default: whichever is newest)")
    p.add_argument("--name", nargs="+", default=None,
                   help="experiments to evaluate (default: every one the run "
                        "directory's config.json lists)")
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
                   help="which splits to evaluate on; train is read off the "
                        "trees' stored leaf counts and costs next to nothing, "
                        "test is routed through every tree")
    p.add_argument("--stride", type=int, default=1,
                   help="evaluate every k-th stored record")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                   help="worker processes, shared by every experiment being "
                        "evaluated (default: one per core)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-evaluate experiments whose .npz already exists")
    args = p.parse_args()

    if args.stride < 1:
        raise SystemExit(f"--stride must be at least 1, got {args.stride}")

    run_id = args.run_id or latest_run_id(args.results_root)
    if run_id is None:
        raise SystemExit(f"no run directories under {args.results_root} -- "
                         f"run examples/incremental_decision_tree/sampler_diagnostics.py first")
    results_dir = os.path.join(args.results_root, run_id)
    if not args.run_id:
        print(f"--run-id not given; using the newest one, {run_id!r}")

    jobs = plan(results_dir, args.name, tuple(args.splits), args.stride,
                args.overwrite)
    if not jobs:
        raise SystemExit(f"nothing to evaluate in {results_dir}")

    print(f">>> {len(jobs)} experiment(s), splits {'+'.join(args.splits)}, "
          f"stride {args.stride}")
    started = time.perf_counter()
    # One bar per job, all drawn by this process in plan order and kept open
    # until the end: closing a bar with leave=True mid-run reprints it at the
    # cursor and shifts the rows of every bar below it.
    bars = [tqdm(total=None, desc=os.path.basename(job[0]), unit="tree",
                 position=i, leave=True)
            for i, job in enumerate(jobs)]
    try:
        if args.jobs > 1:
            # Before the pool exists, so that forked workers inherit them.
            for dataset in sorted({job[2] for job in jobs}):
                splits_for(dataset, SPLITS)
            with ProcessPoolExecutor(max_workers=args.jobs,
                                     mp_context=worker_context()) as pool:
                try:
                    evaluate_in_pool(pool, args.jobs, jobs, bars)
                except BaseException:
                    # Otherwise leaving the pool waits for every chunk queued.
                    pool.shutdown(cancel_futures=True)
                    raise
        else:
            for job, bar in zip(jobs, bars):
                evaluation = FileEvaluation(job, bar)
                for chunk in evaluation.chunks():
                    evaluation.add(chunk, evaluate_chunk(*chunk))
                evaluation.save()
    finally:
        for bar in bars:
            bar.close()

    print(f"\nEvaluated {len(jobs)} experiment(s) in "
          f"{time.perf_counter() - started:.1f}s. Metrics in: {results_dir}"
          f"\n  python examples/incremental_decision_tree/plot_diagnostics.py --run-id {run_id}")


if __name__ == "__main__":
    main()
