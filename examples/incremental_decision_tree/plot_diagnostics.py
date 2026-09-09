"""
Plot the files examples/incremental_decision_tree/sampler_diagnostics.py (or run_experiments.py) and
examples/incremental_decision_tree/evaluate_results.py write into Results/<run-id>/.

    python examples/incremental_decision_tree/sampler_diagnostics.py --dataset wine --run-id wine_baseline
    python examples/incremental_decision_tree/evaluate_results.py --run-id wine_baseline
    python examples/incremental_decision_tree/plot_diagnostics.py --name wine --run-id wine_baseline

The sampler's own columns (tree size, ESS, the move log) come off the .h5
files; the predictive metrics come off the metrics_*.npz beside them, which is
where evaluate_results.py puts them. So the metric figure needs the run to have
been evaluated -- it says so and skips itself if it has not been, rather than
evaluating anything here: an evaluation is minutes to hours of work and belongs
in the step that is built to be re-run with different splits and strides.

--run-id defaults to whichever run is newest under --results-root, and --name
to every experiment that run's config.json lists, so a plain `python
examples/incremental_decision_tree/plot_diagnostics.py` plots everything in the sweep that was just run
without having to name any of it. --name narrows that: it is the <name> half
of the <name>_<sampler>_<proposal>.h5 files, which is the dataset unless a
run_experiments.py entry overrode it with its own 'name' (see experiments.py).
Naming several plots each of them, and adds one all_move_outcomes.png putting
every configuration of every one of them in a single grid.

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
  {name}_move_outcomes.png
                        the same moves, but with the proposal's screen crossed
                        with what the sampler then did, and the acceptance rate
                        of each move type on its axis. This is where a
                        proposal's work goes -- how much of it the surrogate
                        throws away, how much survives the screen only to be
                        rejected, and (for HINTS) how much is screened out of a
                        sweep that lands anyway. SMC panels read "kept" rather
                        than "accepted": it reweights particles instead of
                        rejecting them, so surviving the proposal is as far as
                        the question goes.
  all_move_outcomes.png the same, with every experiment plotted in one grid.
                        Only when more than one is being plotted.
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
from evaluate_results import BOOKKEEPING, metrics_filename
from results_io import (experiment_datasets, find_experiment_files,
                        latest_run_id, load_experiment_hdf5, read_run_config)
from sampler_diagnostics import DEFAULT_CFG

_SAMPLERS = ("mcmc", "smc")
_PROPOSALS = ("MH", "DA", "HINTS")


def known_names(cfg):
    """
    The experiment names a results directory's config.json lists, for the
    "no such experiment" message below. [] if there is nothing to consult
    (an old results directory, or one hand-populated with .h5 files).
    """
    return sorted(experiment_datasets(cfg, DEFAULT_CFG["dataset"]))


def find_runs(results_dir, name):
    """
    {(sampler, proposal): [run, ...]} for the experiment called `name` in
    this run directory -- the sampler's own per-iteration columns.
    """
    return {key: load_experiment_hdf5(path)
            for key, path in find_experiment_files(results_dir, name).items()}


def find_metrics(results_dir, name):
    """
    {(sampler, proposal): {metric: (n_runs, n_points) array}} for the same
    experiment, off the metrics_*.npz evaluate_results.py wrote. Empty for a
    run that has not been evaluated yet.
    """
    out = {}
    for sampler in _SAMPLERS:
        for proposal in _PROPOSALS:
            path = os.path.join(results_dir,
                                metrics_filename(name, sampler, proposal))
            if os.path.exists(path):
                with np.load(path) as data:
                    out[(sampler, proposal)] = {k: data[k] for k in data.files}
    return out


def find_runs_with_metrics(results_dir, name):
    """
    find_runs, with each run's evaluated metrics folded back into its own
    column dict -- so a run reads as one {column: array} mapping holding both
    what the sampler recorded and what the evaluation produced:

        run['n_nodes']         per iteration, from the .h5
        run['test_accuracy']   per evaluated record, from the .npz
        run['metric_iter']     the iteration each of those came from

    The two are on different axes -- the metrics are only as dense as the
    storing and the --stride left them -- so `metric_iter` is what lines them
    up, e.g. `run['cumulative_time'][run['metric_iter']]` for a metric against
    wallclock. An experiment with no metrics file is returned unchanged rather
    than dropped.
    """
    runs_by_key = find_runs(results_dir, name)
    for key, found in find_metrics(results_dir, name).items():
        runs = runs_by_key.get(key)
        if not runs:
            continue
        columns = {k: v for k, v in found.items() if k not in BOOKKEEPING}
        n_evaluated = min((len(v) for v in columns.values()), default=0)
        if n_evaluated != len(runs):
            # An experiment evaluated before a re-run added or lost a chain.
            print(f"[warning] {name} {key[0]}-{key[1]}: metrics cover "
                  f"{n_evaluated} run(s) but the .h5 holds {len(runs)}; "
                  f"re-run evaluate_results.py --overwrite. Skipping them.")
            continue
        for r, run in enumerate(runs):
            run['metric_iter'] = found['iterations']
            for column, values in columns.items():
                run[column] = values[r]
    return runs_by_key


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


def plot_metric(metrics_by_key, metric, out_path, plt, run_id):
    if not metrics_by_key:
        print(f"no metrics_*.npz for this experiment -- skipping the {metric} "
              f"plot. Evaluate the run first:\n"
              f"    python examples/incremental_decision_tree/evaluate_results.py --run-id {run_id}")
        return

    samplers = sorted({k[0] for k in metrics_by_key})
    fig, axes = plt.subplots(1, len(samplers), figsize=(6.5 * len(samplers), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, sampler in zip(axes, samplers):
        for sampler_, proposal in _ordered(metrics_by_key):
            if sampler_ != sampler:
                continue
            found = metrics_by_key[(sampler, proposal)]
            if metric not in found:
                continue
            # One row per chain / repeat, one column per evaluated record; the
            # x-axis is the iteration each of those came from, which the
            # evaluation recorded rather than the plot recomputing it.
            _band(ax, found['iterations'], found[metric],
                  PROPOSAL_COLOR.get(proposal), proposal)
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


# --------------------------------------------------------------------------- #
# move outcomes, crossed with the sampler's own accept/reject
# --------------------------------------------------------------------------- #
#
# What the proposal did with a move (diagnostics.OUTCOMES) is only half of what
# became of it. DA and HINTS screen a move against a surrogate before the outer
# Metropolis step ever sees it, so a move can pass the screen and still be
# thrown away by the sampler -- and, inside a HINTS sweep, be thrown away by
# the screen while the sweep it belonged to is accepted. Crossing the two is
# what says where a proposal's work actually goes.

FATES = (
    "screened in, accepted",         # survived the screen and the sampler took it
    "accepted, no screen",           # MH, and DA's skip-the-screen path
    "screened in, rejected",         # the screen passed it, the sampler did not
    "screened out, sweep accepted",  # HINTS: the sweep landed, this move did not
    "screened out",
    "rejected, no screen",
    "barred / inadmissible",         # never reached the target at all
    "no move available",             # a "stay": no node could be drawn
)
FATE_COLORS = ("#98df8a", "#2ca02c", "#ff7f0e", "#1f77b4", "#ff9896",
               "#d62728", "#7f7f7f", "#c7c7c7")
# The fates that put a move into the chain's state. This is the numerator of
# every acceptance rate below: a move screened out of an accepted HINTS sweep
# was not accepted -- the sweep was, without it.
ACCEPTED_FATES = (0, 1)


def move_fates(run):
    """
    (len(dg.MOVES), len(FATES)) counts for one MCMC run's move log.

    The join is on the call index: the move log tags every move with the
    sample() call it came from, and MCMC makes exactly one call per iteration
    in order, so that index is also the iteration whose accept/reject settled
    it. MH and DA emit one move per call, HINTS one per block, which is the
    whole reason to count moves rather than iterations.

    SMC has no accept/reject step -- it reweights particles rather than
    rejecting them, so a call that proposed something always becomes the new
    particle. Surviving the proposal is then the whole story, and the three
    "rejected" fates below are structurally empty for it. Everything else,
    including how much of the proposal's work the screen throws away, reads the
    same for both samplers.

    None for a run with no move log (sampled with --no-moves).
    """
    if 'move' not in run:
        return None
    move = np.asarray(run['move'], dtype=np.int64)
    outcome = np.asarray(run['outcome'], dtype=np.int64)
    call = np.asarray(run['call'], dtype=np.int64)
    dsurr = np.asarray(run['dsurr'], dtype=np.float64)
    call_id = np.asarray(run['call_id'], dtype=np.int64)
    call_outcome = np.asarray(run['call_outcome'], dtype=np.int64)

    # A call the proposal gave up on handed the state straight back, and an
    # outer MH step then "accepts" it trivially -- the target and proposal
    # terms cancel. Only a call that actually proposed something can carry a
    # real acceptance, which is why call_outcome is consulted either way, and
    # not just the accept flag.
    took = call_outcome == dg.PROPOSED
    if 'accepted' in run:                       # MCMC: the outer step decided
        accepted = np.asarray(run['accepted']).astype(bool)
        if len(accepted) != len(call_id):
            raise ValueError(
                f"{len(call_id)} sample() calls against {len(accepted)} "
                "recorded iterations: crossing the move log with the outer "
                "accept/reject needs them one per iteration.")
        took = took & accepted[call_id]
    outer = np.zeros(len(call_id), dtype=bool)
    outer[call_id[took]] = True

    acc = outer[call]
    # dsurr is the surrogate ratio the screen used, and nan when there was no
    # screen -- so it is exactly the record of whether this move was screened.
    screened = ~np.isnan(dsurr)
    inner_ok = (outcome == dg.PROPOSED) | (outcome == dg.APPLIED)
    inner_no = outcome == dg.SCREENED_OUT
    dead = (outcome == dg.BARRED) | (outcome == dg.INADMISSIBLE)

    fate = np.full(len(move), -1, dtype=np.int64)
    fate[inner_ok & screened & acc] = 0
    fate[inner_ok & ~screened & acc] = 1
    fate[inner_ok & screened & ~acc] = 2
    fate[inner_no & acc] = 3
    fate[inner_no & ~acc] = 4
    fate[inner_ok & ~screened & ~acc] = 5
    fate[dead] = 6
    fate[outcome == dg.STAY] = 7

    keep = fate >= 0
    flat = np.bincount(move[keep] * len(FATES) + fate[keep],
                       minlength=len(dg.MOVES) * len(FATES))
    return flat.reshape(len(dg.MOVES), len(FATES))


def _fate_panel(ax, table, title, sampler, plt):
    """One method's move-type x fate stacked bars, labelled with the rates."""
    import matplotlib.patheffects as pe

    # SMC never rejects, so "accepted" would overstate what its green means:
    # a move that survived the proposal, not one an outer step chose to take.
    verb = "accepted" if sampler == "mcmc" else "kept"

    totals = table.sum(axis=1)
    shown = [i for i, name in enumerate(dg.MOVES) if totals[i] > 0]
    xs = np.arange(len(shown))
    counts = table[shown]
    tallest = counts.sum(axis=1).max()

    bottom = np.zeros(len(shown), dtype=np.float64)
    for k in range(len(FATES)):
        heights = counts[:, k].astype(np.float64)
        ax.bar(xs, heights, bottom=bottom, color=FATE_COLORS[k],
               edgecolor='black', linewidth=0.6, label=FATES[k])
        for j, height in enumerate(heights):
            share = height / counts[j].sum()
            # Only labels with room to be read: the small slices are what made
            # the stacked percentages illegible, and the bar shows them anyway.
            if share >= 0.03:
                ax.text(xs[j], bottom[j] + height / 2, f"{100 * share:.2f}%",
                        ha='center', va='center', size=9, weight='bold',
                        color=FATE_COLORS[k],
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground='black')])
        bottom += heights

    # A "stay" is not a move that was rejected, it is a call with no move in
    # it, so it carries no acceptance rate and is kept out of the panel's.
    real = [j for j, i in enumerate(shown) if dg.MOVES[i] != "stay"]
    accepted = counts[:, list(ACCEPTED_FATES)].sum(axis=1)
    labels = []
    for j, i in enumerate(shown):
        label = f"{dg.MOVES[i]}\n{100 * totals[i] / totals.sum():.1f}% of calls"
        if j in real:
            label += f"\n{verb} {100 * accepted[j] / counts[j].sum():.2f}%"
        labels.append(label)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, size=9)

    n_moves = counts[real].sum()
    overall = accepted[real].sum() / n_moves if n_moves else 0.0
    ax.set_title(f"{title}\n{int(n_moves):,} moves, "
                 f"{100 * overall:.2f}% {verb}", weight='bold', size=11)
    ax.set_ylabel("moves considered")
    ax.set_ylim(0, tallest * 1.08)
    ax.grid(alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_move_outcomes(panels, out_path, plt):
    """
    One stacked bar chart per MCMC configuration: every move the proposal
    considered, by move type and by what became of it.

    `panels` is [(title, sampler, [run, ...]), ...]; the runs of one
    configuration are summed, since a chain (or an SMC repeat) is a repeat
    rather than a separate condition.

    Both samplers appear, but they answer slightly different questions: for
    MCMC the green fates are moves an outer Metropolis step chose to take, for
    SMC they are moves that survived the proposal, which SMC then reweights
    rather than accepting. The panels say which, and move_fates explains why.
    """
    tables = []
    for title, sampler, runs in panels:
        found = [move_fates(run) for run in runs]
        found = [t for t in found if t is not None]
        if found:
            tables.append((title, sampler, np.sum(found, axis=0)))
    if not tables:
        print("no move log in any run -- skipping the move outcome plot "
              "(the harness was run with --no-moves)")
        return

    cols = min(3, len(tables))
    rows = int(np.ceil(len(tables) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 6.0 * rows),
                             squeeze=False, layout='constrained')
    flat = axes.flatten()
    for ax, (title, sampler, table) in zip(flat, tables):
        _fate_panel(ax, table, title, sampler, plt)
    for ax in flat[len(tables):]:
        ax.set_visible(False)

    # Built from the fixed colour table rather than collected off one panel:
    # which fates appear differs by proposal (MH never screens, HINTS never
    # reaches the outer step with an unscreened move), so no single panel's
    # handles cover every colour in use.
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, edgecolor='black', label=name)
                        for name, c in zip(FATES, FATE_COLORS)],
               loc='outside upper center', ncol=4, fontsize=9)
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
    p.add_argument("--name", nargs="+", default=None,
                   help="experiments to plot (default: every one this run's "
                        "config.json lists). Naming several plots each of "
                        "them and puts all their configurations in one "
                        "move outcome grid, for comparing settings side by "
                        "side")
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
                         f"run examples/incremental_decision_tree/sampler_diagnostics.py first")
    results_dir = os.path.join(args.results_root, run_id)
    if not args.run_id:
        print(f"--run-id not given; using the newest one, {run_id!r}")

    # What the run actually holds, which is what to plot when nothing is named.
    # The dataset default is only a fallback for a results directory written
    # before config.json, or one hand-populated with .h5 files.
    known = known_names(read_run_config(results_dir))
    names = args.name or known or [DEFAULT_CFG["dataset"]]

    out_dir = args.out_dir or os.path.join(results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    everything, plotted = [], []
    for name in names:
        runs_by_key = find_runs(results_dir, name)
        if not runs_by_key:
            hint = f" (this run has: {', '.join(known)})" if known else ""
            print(f"no experiment named {name!r} in {results_dir}{hint} -- skipping")
            continue
        plotted.append(name)

        plot_metric(find_metrics(results_dir, name), args.metric,
                    os.path.join(out_dir, f"{name}_metric.png"), plt, run_id)
        plot_tree_size(runs_by_key,
                       os.path.join(out_dir, f"{name}_tree_size.png"), plt)
        plot_moves(runs_by_key,
                   os.path.join(out_dir, f"{name}_moves.png"), plt)

        # _ordered groups by sampler and then MH -> DA -> HINTS, so with
        # cols=3 each sampler gets a row and a proposal keeps its column.
        panels = [(f"{sampler.upper()}-{proposal}", sampler,
                   runs_by_key[(sampler, proposal)])
                  for sampler, proposal in _ordered(runs_by_key)]
        plot_move_outcomes(panels,
                           os.path.join(out_dir, f"{name}_move_outcomes.png"),
                           plt)
        everything += [(f"{name}\n{title}", sampler, runs)
                       for title, sampler, runs in panels]

        plot_ess(runs_by_key, os.path.join(out_dir, f"{name}_ess.png"), plt)

    if not plotted:
        raise SystemExit(f"nothing to plot in {results_dir}")
    if len(plotted) > 1:
        # Every experiment's configurations in one grid: with the experiments
        # in the rows this is the cross-setting comparison, which no per-name
        # figure can show.
        plot_move_outcomes(everything,
                           os.path.join(out_dir, "all_move_outcomes.png"), plt)


if __name__ == "__main__":
    main()
