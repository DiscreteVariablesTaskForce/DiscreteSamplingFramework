"""
Plot the .h5 files examples/sampler_diagnostics.py (or run_experiments.py)
writes into Results/<run-id>/.

    python examples/sampler_diagnostics.py --dataset wine --run-id wine_baseline
    python examples/plot_diagnostics.py --name wine --run-id wine_baseline

--run-id defaults to whichever run is newest under --results-root, so a plain
`python examples/plot_diagnostics.py` plots the sweep that was just run
without having to name it. --name is the experiment to plot: the <name> half
of the <name>_<sampler>_<proposal>.h5 files it looks for, which is the dataset
unless a run_experiments.py entry overrode it with its own 'name' (see
experiments.py); --name defaults to "wine", sampler_diagnostics.py's own
default dataset, which is what makes both scripts' bare defaults line up.

Every proposal (and both samplers) found for --name in the run directory is
overlaid on the same axes, so the point of each figure is the comparison
between them:

  {name}_metric.png     --metric against iteration (MCMC) / step (SMC), mean
                        over chains with the interquartile range shaded
  {name}_tree_size.png  tree size, same layout
  {name}_moves.png      move type x outcome as a stacked bar per proposal --
                        MoveLog.format_counts()'s table, drawn instead of
                        tabulated. Skipped if a run has no move log (the
                        harness's --no-moves).
  {name}_ess.png        SMC only: effective sample size against step

A run's own iteration/step axis is trusted rather than recomputed: it comes
from the same --metric-every the harness recorded with, so chains from
different harness invocations plot correctly side by side even if their
settings differed.
"""
import argparse
import os

import numpy as np

from discretesampling.domain.incremental_decision_tree import diagnostics as dg
from results_io import latest_run_id, load_experiment_hdf5, read_run_config
from sampler_diagnostics import DEFAULT_CFG

_SAMPLERS = ("mcmc", "smc")
_PROPOSALS = ("MH", "DA", "HINTS")


def known_names(cfg):
    """
    The experiment names a results directory's config.json lists, for the
    "no such experiment" message below. [] if there is nothing to consult
    (an old results directory, or one hand-populated with .h5 files).
    """
    if "experiments" in cfg:
        return sorted({e.get("name") or e.get("dataset", DEFAULT_CFG["dataset"])
                       for e in cfg["experiments"]})
    if cfg.get("dataset"):
        return [cfg["dataset"]]
    return []


def find_runs(results_dir, name):
    """
    {(sampler, proposal): [run, ...]} for the experiment called `name` in
    this run directory: <name>_<sampler>_<proposal>.h5, checked directly
    against the fixed samplers x proposals set.

    Not globbed on the name as a prefix: two experiment names can share one
    (dataset "wine" and a second experiment named "wine_deep", say), and a
    glob like "wine_*_*.h5" would match both -- "wine_deep_mcmc_HINTS.h5" is
    indistinguishable from a hypothetical "wine" experiment using a sampler
    called "deep_mcmc" from the filename's shape alone. Checking the exact,
    fixed set of (sampler, proposal) pairs sidesteps the ambiguity rather than
    trying to parse around it.
    """
    out = {}
    for sampler in _SAMPLERS:
        for proposal in _PROPOSALS:
            path = os.path.join(results_dir, f"{name}_{sampler}_{proposal}.h5")
            if os.path.exists(path):
                out[(sampler, proposal)] = load_experiment_hdf5(path)
    return out


PROPOSAL_ORDER = ["MH", "DA", "HINTS"]
PROPOSAL_COLOR = dict(zip(PROPOSAL_ORDER, ["#4c72b0", "#dd8452", "#55a868"]))


def _ordered(keys):
    return sorted(keys, key=lambda k: (k[0], PROPOSAL_ORDER.index(k[1])
                                       if k[1] in PROPOSAL_ORDER else 99))


def _band(ax, x, y, color, label):
    """Mean over runs, IQR shaded -- the one plot idiom every figure below uses."""
    lo, hi = np.percentile(y, [25, 75], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)
    ax.plot(x, y.mean(axis=0), color=color, lw=1.4, label=label)


def plot_metric(runs_by_key, metric, out_path, plt):
    samplers = sorted({k[0] for k in runs_by_key})
    fig, axes = plt.subplots(1, len(samplers), figsize=(6.5 * len(samplers), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, sampler in zip(axes, samplers):
        for sampler_, proposal in _ordered(runs_by_key):
            if sampler_ != sampler:
                continue
            runs = runs_by_key[(sampler, proposal)]
            if metric not in runs[0]:
                continue
            x = runs[0]['metric_iter']
            y = np.vstack([r[metric] for r in runs])
            _band(ax, x, y, PROPOSAL_COLOR.get(proposal), proposal)
        ax.set_xlabel("iteration" if sampler == "mcmc" else "step")
        ax.set_ylabel(metric)
        ax.set_title(sampler.upper())
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_tree_size(runs_by_key, out_path, plt):
    samplers = sorted({k[0] for k in runs_by_key})
    fig, axes = plt.subplots(1, len(samplers), figsize=(6.5 * len(samplers), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, sampler in zip(axes, samplers):
        size_key = 'n_nodes' if sampler == 'mcmc' else 'mean_nodes'
        for sampler_, proposal in _ordered(runs_by_key):
            if sampler_ != sampler:
                continue
            runs = runs_by_key[(sampler, proposal)]
            y = np.vstack([r[size_key] for r in runs])
            x = np.arange(y.shape[1])
            _band(ax, x, y, PROPOSAL_COLOR.get(proposal), proposal)
        ax.set_xlabel("iteration" if sampler == "mcmc" else "step")
        ax.set_ylabel("tree size (nodes)" if sampler == "mcmc" else "mean tree size across particles")
        ax.set_title(sampler.upper())
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_moves(runs_by_key, out_path, plt):
    from matplotlib.patches import Patch

    keys = [k for k in _ordered(runs_by_key) if 'move' in runs_by_key[k][0]]
    if not keys:
        print("no move log in any run -- skipping moves plot "
              "(the harness was run with --no-moves)")
        return

    samplers = sorted({k[0] for k in keys})
    fig, axes = plt.subplots(len(samplers), 3, figsize=(13, 4.5 * len(samplers)),
                             squeeze=False)

    outcome_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(dg.OUTCOMES)))
    move_names = [m for m in dg.MOVES if m != "stay"]  # a "stay" has no other outcome

    for row, sampler in enumerate(samplers):
        cols = [k for k in keys if k[0] == sampler]
        for ax, key in zip(axes[row], cols):
            proposal = key[1]
            runs = runs_by_key[key]
            move = np.concatenate([r['move'] for r in runs])
            outcome = np.concatenate([r['outcome'] for r in runs])
            flat = np.bincount(move.astype(np.int64) * len(dg.OUTCOMES)
                               + outcome.astype(np.int64),
                               minlength=len(dg.MOVES) * len(dg.OUTCOMES))
            table = flat.reshape(len(dg.MOVES), len(dg.OUTCOMES))

            bottoms = np.zeros(len(move_names))
            xs = np.arange(len(move_names))
            for oi, oname in enumerate(dg.OUTCOMES):
                heights = np.array([table[dg.MOVE_CODE[m], oi] for m in move_names])
                if heights.sum() == 0:
                    continue
                ax.bar(xs, heights, bottom=bottoms, color=outcome_colors[oi],
                       label=oname)
                bottoms += heights
            ax.set_xticks(xs)
            ax.set_xticklabels(move_names)
            ax.set_title(f"{sampler.upper()}-{proposal}")
            ax.set_ylabel("moves considered")
        for ax in axes[row][len(cols):]:
            ax.set_visible(False)

    # Built from the fixed colour table rather than collected off one subplot:
    # which outcomes appear differs by proposal (MH never screens, HINTS never
    # "proposes"), so no single subplot's handles cover every colour in use.
    handles = [Patch(color=outcome_colors[oi], label=oname)
               for oi, oname in enumerate(dg.OUTCOMES) if oname != "stay"]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_ess(runs_by_key, out_path, plt):
    keys = [k for k in _ordered(runs_by_key) if k[0] == "smc"]
    if not keys:
        print("no SMC runs found -- skipping ESS plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, proposal in keys:
        runs = runs_by_key[("smc", proposal)]
        y = np.vstack([r['ess_history'] for r in runs])
        x = np.arange(y.shape[1])
        _band(ax, x, y, PROPOSAL_COLOR.get(proposal), proposal)
        resample_rate = np.mean([r['resampled_history'].mean() for r in runs])
        ax.plot([], [], ' ', label=f"{proposal}: resampled {resample_rate:.0%} of steps")

    n_particles = keys and runs_by_key[keys[0]][0]['ess'].max()
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("effective sample size")
    ax.set_title(f"SMC effective sample size (of {int(n_particles)} particles)"
                 if n_particles else "SMC effective sample size")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=DEFAULT_CFG["dataset"],
                   help="experiment to plot -- the dataset, unless a "
                        "run_experiments.py entry overrode 'name'")
    p.add_argument("--results-root", default="Results")
    p.add_argument("--run-id", default=None,
                   help="results directory under --results-root "
                        "(default: whichever is newest)")
    p.add_argument("--out-dir", default=None,
                   help="default: <results-root>/<run-id>/plots")
    p.add_argument("--metric", default="test_accuracy",
                   help="any column classification_metrics produces, e.g. "
                        "test_macro_f1, test_log_loss, train_accuracy")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required to plot diagnostics")

    run_id = args.run_id or latest_run_id(args.results_root)
    if run_id is None:
        raise SystemExit(f"no run directories under {args.results_root} -- "
                         f"run examples/sampler_diagnostics.py first")
    results_dir = os.path.join(args.results_root, run_id)
    if not args.run_id:
        print(f"--run-id not given; using the newest one, {run_id!r}")

    runs_by_key = find_runs(results_dir, args.name)
    if not runs_by_key:
        names = known_names(read_run_config(results_dir))
        hint = f" (this run has: {', '.join(names)})" if names else ""
        raise SystemExit(f"no experiment named {args.name!r} in {results_dir}{hint}")

    out_dir = args.out_dir or os.path.join(results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_metric(runs_by_key, args.metric,
                os.path.join(out_dir, f"{args.name}_metric.png"), plt)
    plot_tree_size(runs_by_key,
                   os.path.join(out_dir, f"{args.name}_tree_size.png"), plt)
    plot_moves(runs_by_key,
               os.path.join(out_dir, f"{args.name}_moves.png"), plt)
    plot_ess(runs_by_key,
             os.path.join(out_dir, f"{args.name}_ess.png"), plt)


if __name__ == "__main__":
    main()
