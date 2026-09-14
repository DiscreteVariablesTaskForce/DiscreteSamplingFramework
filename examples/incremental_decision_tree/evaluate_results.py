"""
Turn the states a sweep stored into predictive metrics.

sampler_diagnostics.py samples and stores trees; this reads them back and
evaluates them, writing one metrics_<experiment>.npz beside each .h5:

    python examples/incremental_decision_tree/sampler_diagnostics.py --dataset wine --run-id wine_baseline
    python examples/incremental_decision_tree/evaluate_results.py --run-id wine_baseline
    python examples/incremental_decision_tree/plot_diagnostics.py --run-id wine_baseline

    python examples/incremental_decision_tree/evaluate_results.py --run-id covtype --name covtype_mcmc \
        --stride 10 --splits test --jobs 8

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

The .npz holds one (n_runs, n_points) array per metric -- the runs are the
chains (MCMC) or the repeats (SMC) -- plus `iterations`, the sampler iteration
or step behind each point, and the confusion matrices as (n_runs, n_points,
K, K). plot_diagnostics.py reads exactly this.
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discretesampling.domain import incremental_decision_tree as idt  # noqa: E402
from results_io import (experiment_datasets, find_experiment_files,  # noqa: E402
                        latest_run_id, load_experiment_hdf5, read_run_config,
                        states_of)
from sampler_diagnostics import DEFAULT_CFG, load_data  # noqa: E402

METRICS_PREFIX = "metrics_"
SPLITS = ("train", "test")

# Bookkeeping written alongside the metrics, but not metrics themselves: they
# describe the columns rather than measuring anything, so `for metric in npz`
# style reads have to be able to tell them apart.
BOOKKEEPING = ("iterations", "stride", "n_runs")


def metrics_filename(name, sampler, proposal):
    return f"{METRICS_PREFIX}{name}_{sampler}_{proposal}.npz"


# --------------------------------------------------------------------------- #
# evaluating one run
# --------------------------------------------------------------------------- #

# Loaded once per process rather than once per run: a worker evaluates several
# runs of the same experiment, and re-splitting covtype for each of them costs
# more than the evaluation.
_DATA = {}


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


def evaluate_series(series, ensembles, splits, pbar=None):
    """
    {metric: list over records} for one run, given the (slots, weights) of the
    records to evaluate, from picked_ensembles.

    A record's metrics are the weighted mean of its trees' own metrics, and a
    tree's metrics don't depend on the weight it carries, so each distinct
    stored tree is evaluated once and reused by every record holding it: an
    MCMC chain sitting on a rejected state and an SMC particle surviving from
    step to step cost nothing after the first time. `pbar`, if given, ticks
    once per tree evaluated, which is where the time goes.
    """
    per_tree = {}
    cols = defaultdict(list)
    for slots, weights in ensembles:
        for s in slots:
            if s not in per_tree:
                per_tree[s] = tree_metrics(series.state(s), splits)
                if pbar is not None:
                    pbar.update(1)
        mean = idt.weighted_mean([per_tree[s] for s in slots], weights)
        for k, v in mean.items():
            cols[k].append(v)
    return dict(cols), len(per_tree)


def evaluate_file(path, dataset, wanted_splits, stride, position=0):
    """
    Every run in one experiment's .h5, as {metric: (n_runs, n_points) array}
    plus the iteration behind each point.

    `position` slots this file's progress bar into its own terminal row, so
    several files evaluated in parallel (--jobs > 1) don't overwrite each
    other's line.
    """
    splits = splits_for(dataset, wanted_splits)
    runs = load_experiment_hdf5(path)

    series_of_run = []
    for run in runs:
        series = states_of(run)
        if series is None:
            raise SystemExit(
                f"{path} holds no states: it was sampled with --no-states, and "
                f"there is nothing to evaluate. Re-run the sampling without it.")
        series_of_run.append(series)

    if series_of_run:
        [(_, X_fit, y_fit)] = splits_for(dataset, ("train",))
        check_fitted_to(path, series_of_run[0], X_fit, y_fit)

    ensembles = [picked_ensembles(series, stride) for series in series_of_run]
    total = sum(len({int(s) for slots, _ in picked for s in slots})
                for picked in ensembles)
    pbar = tqdm(total=total, desc=os.path.basename(path), unit="tree",
                position=position, leave=False)

    per_run, evaluated = [], 0
    for series, picked in zip(series_of_run, ensembles):
        cols, n_evaluated = evaluate_series(series, picked, splits, pbar)
        per_run.append(cols)
        evaluated += n_evaluated
    pbar.close()

    # Runs of one experiment share iters/steps and store_every, so they line up;
    # a run cut short (a crash mid-sweep) is truncated to the common length
    # rather than refused, so the rest of the sweep still plots.
    lengths = {len(next(iter(c.values()))) for c in per_run if c}
    n_points = min(lengths) if lengths else 0
    if len(lengths) > 1:
        print(f"    [warning] runs hold {sorted(lengths)} evaluated records; "
              f"truncating to {n_points}")
    iterations = series_of_run[0].record_iterations()[::stride][:n_points]

    keys = sorted(set().union(*(c.keys() for c in per_run))) if per_run else []
    matrices = {k: np.stack([np.asarray(c[k][:n_points]) for c in per_run])
                for k in keys}
    return matrices, iterations, evaluated


def evaluate_one(task):
    """One (experiment file -> .npz) job, as a worker sees it."""
    path, out_path, dataset, wanted_splits, stride, position = task
    started = time.perf_counter()
    matrices, iterations, evaluated = evaluate_file(
        path, dataset, wanted_splits, stride, position)
    np.savez_compressed(out_path, iterations=iterations, stride=stride,
                        n_runs=len(next(iter(matrices.values()))) if matrices else 0,
                        **matrices)
    return out_path, matrices, evaluated, time.perf_counter() - started


# --------------------------------------------------------------------------- #
# driving a whole results directory
# --------------------------------------------------------------------------- #

def plan(results_dir, names, wanted_splits, stride, overwrite):
    """The (path, out_path, dataset, splits, stride) jobs for a run directory."""
    datasets = experiment_datasets(read_run_config(results_dir),
                                   DEFAULT_CFG["dataset"])
    if names:
        unknown = [n for n in names if n not in datasets]
        if unknown and datasets:
            raise SystemExit(f"no experiment named {', '.join(unknown)} in "
                             f"{results_dir} (it has: {', '.join(sorted(datasets))})")
    else:
        names = sorted(datasets)

    jobs = []
    for name in names:
        for (sampler, proposal), path in find_experiment_files(results_dir, name).items():
            out_path = os.path.join(results_dir,
                                    metrics_filename(name, sampler, proposal))
            if os.path.exists(out_path) and not overwrite:
                print(f">>> Skipping {os.path.basename(path)}: "
                      f"{os.path.basename(out_path)} exists (--overwrite to redo it)")
                continue
            jobs.append((path, out_path, datasets[name], wanted_splits, stride))
    return jobs


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
    print(line, flush=True)


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
    p.add_argument("--jobs", type=int, default=DEFAULT_CFG["jobs"],
                   help="experiments to evaluate at once")
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
    if args.jobs > 1 and len(jobs) > 1:
        # Positions round-robin over the worker slots so concurrent files'
        # progress bars land on their own terminal row instead of clobbering
        # each other; it's an approximation of which task lands on which
        # worker, which is fine for a progress bar.
        max_workers = min(args.jobs, len(jobs))
        tasks = [job + (i % max_workers,) for i, job in enumerate(jobs)]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(evaluate_one, task) for task in tasks]
            for future in as_completed(futures):
                summarise(*future.result())
    else:
        for job in jobs:
            summarise(*evaluate_one(job + (0,)))

    print(f"\nEvaluated {len(jobs)} experiment(s) in "
          f"{time.perf_counter() - started:.1f}s. Metrics in: {results_dir}"
          f"\n  python examples/incremental_decision_tree/plot_diagnostics.py --run-id {run_id}")


if __name__ == "__main__":
    main()
