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

It is also much cheaper than the inline version was, for two reasons that are
both exact rather than approximations:

  * A record is evaluated once per *distinct ensemble*. An MCMC chain that
    rejects sits on one state for many iterations and a resampled SMC step
    holds one tree in many slots; both collapse before anything is routed.
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
    """[(prefix, X, y), ...] for the named splits of a dataset."""
    if dataset not in _DATA:
        X_train, X_test, y_train, y_test = load_data(dataset)
        _DATA[dataset] = {"train": ("train_", X_train, y_train),
                          "test": ("test_", X_test, y_test)}
    return [_DATA[dataset][name] for name in wanted]


def evaluate_series(series, splits, stride=1):
    """
    {metric: list over evaluated records} for one run's stored states.

    Records that hold an ensemble already evaluated are read off the cache
    rather than routed again. For MCMC that is every rejected iteration -- the
    chain is on the state it was on -- and it also covers a chain returning to
    a tree it left. For SMC it needs the weights to repeat too, which they
    rarely do, so there the saving is the within-step dedup ensemble() does.
    """
    cache = {}
    cols = defaultdict(list)
    picks = range(0, len(series), stride)
    for i in picks:
        slots, weights = series.ensemble(i)
        key = (slots.tobytes(), weights.tobytes())
        out = cache.get(key)
        if out is None:
            states = [series.state(s) for s in slots]
            out = {}
            for prefix, X, y in splits:
                out.update(idt.evaluate(states, X, y, weights=weights,
                                        num_classes=series.num_classes,
                                        prefix=prefix))
            cache[key] = out
        for k, v in out.items():
            cols[k].append(v)
    iterations = series.record_iterations()[list(picks)]
    return dict(cols), iterations, len(cache)


def evaluate_file(path, dataset, wanted_splits, stride):
    """
    Every run in one experiment's .h5, as {metric: (n_runs, n_points) array}
    plus the iteration behind each point.
    """
    splits = splits_for(dataset, wanted_splits)
    runs = load_experiment_hdf5(path)

    per_run, iterations, evaluated = [], None, 0
    for run in runs:
        series = states_of(run)
        if series is None:
            raise SystemExit(
                f"{path} holds no states: it was sampled with --no-states, and "
                f"there is nothing to evaluate. Re-run the sampling without it.")
        cols, iters, n_evaluated = evaluate_series(series, splits, stride)
        per_run.append(cols)
        evaluated += n_evaluated
        iterations = iters if iterations is None else iterations

    # Runs of one experiment share iters/steps and store_every, so they line up;
    # a run cut short (a crash mid-sweep) is truncated to the common length
    # rather than refused, so the rest of the sweep still plots.
    lengths = {len(next(iter(c.values()))) for c in per_run if c}
    n_points = min(lengths) if lengths else 0
    if len(lengths) > 1:
        print(f"    [warning] runs hold {sorted(lengths)} evaluated records; "
              f"truncating to {n_points}")
    iterations = iterations[:n_points]

    keys = sorted(set().union(*(c.keys() for c in per_run))) if per_run else []
    matrices = {k: np.stack([np.asarray(c[k][:n_points]) for c in per_run])
                for k in keys}
    return matrices, iterations, evaluated


def evaluate_one(task):
    """One (experiment file -> .npz) job, as a worker sees it."""
    path, out_path, dataset, wanted_splits, stride = task
    started = time.perf_counter()
    matrices, iterations, evaluated = evaluate_file(
        path, dataset, wanted_splits, stride)
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
    line = (f"[{os.path.basename(out_path)}] {evaluated} distinct ensembles "
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
                   help="which splits to evaluate on; dropping 'train' roughly "
                        "halves the cost")
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
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            futures = [pool.submit(evaluate_one, job) for job in jobs]
            for future in as_completed(futures):
                summarise(*future.result())
    else:
        for job in jobs:
            summarise(*evaluate_one(job))

    print(f"\nEvaluated {len(jobs)} experiment(s) in "
          f"{time.perf_counter() - started:.1f}s. Metrics in: {results_dir}"
          f"\n  python examples/incremental_decision_tree/plot_diagnostics.py --run-id {run_id}")


if __name__ == "__main__":
    main()
