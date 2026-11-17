"""
The loading and plotting code from the Results_Viewer notebooks, in one place.

Everything here reads what the experiment pipeline wrote to Results/<run-id>/:
the <stem>_metrics.npz files evaluate_results.py produced, and the <stem>.h5
files the sampler wrote (runtime, ESS, stored tree sizes, the move log).
Paths are relative to the repo root, which is where the notebooks run from.

From a notebook at the repo root:

    import sys
    sys.path.insert(0, "examples/incremental_decision_tree")
    import results_viewer as rv

    smc_results, mcmc_results = rv.load_results("261113_132259")
    smc_combined = rv.combine_results(smc_results)
    rv.plot_metric_split(smc_combined, "test_accuracy", "Test Accuracy", "SMC", "Accuracy")

Each experiment is a plain dict (load_metrics_files lists the keys). The plot
functions call plt.show() themselves and return nothing. The table builders
return their tables.

Three orderings are used, matching the notebooks:
  - combine_results: DA, Flat HINTS, HINTS, MH, largest subsample first.
    Used by plot_metric_split.
  - by_proposal: HINTS, Flat HINTS, DA, MH. Used by the per-experiment grids
    (runtime, ESS, chains) and the cap analysis.
  - mh_last: proposals alphabetically, MH last. Used by the move outcome,
    move fate and outcome trace plots.
"""
import json
import os

import h5py
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from discretesampling.domain.incremental_decision_tree.diagnostics import (
    APPLIED, BARRED, INADMISSIBLE, MOVE_CODE, MOVES, OUTCOMES, PROPOSED, SCREENED_OUT, STAY)
from plot_diagnostics import ACCEPTED_FATES, move_fates
from results_io import (FULL_DATA_PROPOSALS, PRIOR_INIT, PROPOSALS, RUN_PREFIX, experiment_stem,
                        init_of)


def set_style():
    """The look every notebook set in its first cell."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'text.usetex': False})


set_style()


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

# The name each results_io.PROPOSALS entry is shown under. Files for a proposal
# not listed in results_io.PROPOSALS (e.g. an old HINTS proxy) are skipped.
PROPOSAL_LABELS = {"MH": "MH", "DA": "DA", "FlatHINTS": "Flat HINTS", "HINTS": "HINTS"}


def parse_ss(tag):
    """The {ss} field of a filename back to ss_prop: '12p5' -> 0.125, '50' -> 0.5."""
    return float(tag.replace("p", ".")) / 100


def load_config(run_id, results_root="Results"):
    """The config.json written next to a run's .h5 files."""
    with open(os.path.join(results_root, run_id, "config.json")) as f:
        return json.load(f)


def _as_list(value):
    return [value] if isinstance(value, str) else list(value)


def config_stems(entry):
    """{stem: (sampler, proposal)} for the files one config.json entry's runs
    were written under. One entry can list several samplers and proposals,
    and each (sampler, proposal) pair has its own files. Names are built by
    results_io.experiment_stem, so they always match what was written."""
    if "dataset" not in entry:
        return {}
    init = init_of(entry)
    stems = {}
    for sampler in _as_list(entry.get("samplers", ())):
        for proposal in _as_list(entry.get("proposals", ())):
            if proposal not in PROPOSALS:
                continue
            ss_prop = entry.get("ss_prop")
            if proposal in FULL_DATA_PROPOSALS:
                ss_prop = 1.0
            elif ss_prop is None:
                continue
            stem = experiment_stem(entry["dataset"], sampler, proposal, ss_prop, init)
            stems[stem] = (sampler, proposal)
    return stems


def load_metrics_files(run_id, results_root="Results"):
    """
    (smc, mcmc) lists of experiment dicts, one per *_metrics.npz in
    Results/<run_id> that config.json has an experiment for. A metrics file
    that no config entry accounts for is skipped, and the skipped files are
    printed.

    Keys of each experiment:
        name, run_id, dataset, sampler, proposal_mechanism (the display name)
        ss_prop          subsample proportion. None for MH. label_hierarchies
                         later rewrites it for (Flat) HINTS
        init_from_prior  True if every chain / particle drew its own starting
                         tree from the prior
        init_tree_size   size of a random starting tree (0 for a stump, and
                         for starts drawn from the prior)
        init_tree_name   name of a given starting tree, else None
        init_tree        the starting tree's rows if one was given, else None
        path, h5_path    the metrics .npz and the .h5 it was evaluated from
        stride           every stride-th stored record was evaluated
        data             the arrays in the .npz
        chains, particles (None for MCMC), max_tree_size   from config.json
    """
    results_path = os.path.join(results_root, run_id)
    config = load_config(run_id, results_root)
    # run_experiments.py writes a list of experiments. A single
    # sampler_diagnostics.py sweep writes the one config it ran.
    entries = config.get("experiments") or ([config] if "dataset" in config else [])
    known = {stem: (entry, sampler, proposal)
             for entry in entries
             for stem, (sampler, proposal) in config_stems(entry).items()}

    smc, mcmc, skipped = [], [], []
    for filename in sorted(os.listdir(results_path)):
        if not filename.endswith("_metrics.npz"):
            continue
        stem = filename[:-len("_metrics.npz")]
        if stem not in known:
            skipped.append(filename)
            continue
        entry, sampler, proposal = known[stem]
        init = init_of(entry)

        file_path = os.path.join(results_path, filename)
        with np.load(file_path) as npz:
            # np.load is lazy and holds the file open, so read everything now.
            data = {key: npz[key] for key in npz.files}

        experiment = {
            "name": stem,
            "run_id": run_id,
            "dataset": entry["dataset"],
            # MH runs on the full data. None keeps it apart from the subsampled runs.
            "ss_prop": None if proposal in FULL_DATA_PROPOSALS else float(entry["ss_prop"]),
            "init_from_prior": init == PRIOR_INIT,
            "init_tree_size": init if isinstance(init, int) else 0,
            "init_tree_name": init if isinstance(init, str) and init != PRIOR_INIT else None,
            "sampler": sampler,
            "proposal_mechanism": PROPOSAL_LABELS.get(proposal, proposal),
            "path": file_path,
            "h5_path": os.path.join(results_path, stem + ".h5"),
            # data['iterations'] already holds the iteration / step each
            # evaluated point came from, so plots use that, not the point's index.
            "stride": int(data["stride"]) if "stride" in data else 1,
            "data": data,
            "init_tree": entry.get("init_tree"),
            "chains": entry.get("chains"),
            "particles": entry.get("particles") if sampler == "smc" else None,
            "max_tree_size": entry.get("max_tree_size"),
        }
        (smc if sampler == "smc" else mcmc).append(experiment)

    if skipped:
        print(f"Skipped {len(skipped)} metrics file(s) with no matching experiment "
              f"in config.json: {', '.join(skipped)}")
    return smc, mcmc


def _run_names(f):
    """The Run_N groups of an open .h5 file, in run order."""
    return sorted((k for k in f if k.startswith(RUN_PREFIX)),
                  key=lambda k: int(k[len(RUN_PREFIX):]))


def record_positions(experiment):
    """Which stored records were evaluated: every stride-th one. state_index
    is indexed by record, while cumulative_time and ess are indexed by
    iteration. The two only line up when stride and store_every are both 1."""
    return np.arange(len(experiment['data']['iterations'])) * experiment['stride']


def load_n_iters(experiment):
    """Iterations (MCMC) or steps (SMC) the run actually took."""
    with h5py.File(experiment['h5_path'], "r") as f:
        return len(f[_run_names(f)[0]]["cumulative_time"])


def _load_per_iteration(experiment, dataset):
    """An (n_runs, n_points) array of a per-iteration .h5 dataset, read at the
    iterations the metrics were evaluated at."""
    iterations = experiment['data']['iterations']
    n_runs = int(experiment['data']['n_runs'])
    with h5py.File(experiment['h5_path'], "r") as f:
        return np.stack([f[run][dataset][()][iterations] for run in _run_names(f)[:n_runs]])


def load_runtime(experiment):
    """Cumulative sampling time in seconds, as an (n_runs, n_points) array
    lined up with the metric arrays. The clock times the sampler only: time
    spent recording states is left out."""
    return _load_per_iteration(experiment, "cumulative_time")


def load_ess(experiment):
    """Effective sample size per step, as an (n_runs, n_points) array lined
    up with the metric arrays. SMC only."""
    return _load_per_iteration(experiment, "ess")


def load_tree_len(experiment, particles="mean"):
    """
    Tree size in internal nodes, as an (n_runs, n_points) array lined up with
    the metric arrays.

    state_tree_lengths has one entry per *stored* tree, not per iteration,
    because consecutive identical states are stored once. state_index maps an
    iteration (and, for SMC, a particle) to a slot in that store.

    SMC has one tree per particle, so `particles` picks how to reduce them:
    "mean", "max", or "all" to keep the full (n_runs, n_points, n_particles)
    array. All three give the same result for MCMC.
    """
    reducers = {"mean": lambda a: a.mean(axis=1),
                "max": lambda a: a.max(axis=1),
                "all": lambda a: a}
    if particles not in reducers:
        raise ValueError(f"particles must be one of {sorted(reducers)}, got {particles!r}")
    reduce = reducers[particles]
    n_runs = int(experiment['data']['n_runs'])
    positions = record_positions(experiment)
    with h5py.File(experiment['h5_path'], "r") as f:
        return np.stack([
            reduce(f[run]["state_tree_lengths"][()][f[run]["state_index"][()][positions]])
            for run in _run_names(f)[:n_runs]])


def attach_series(results):
    """Add the .h5 series the plots need to each experiment: 'time',
    'tree_len' and 'n_iters', plus 'ess' for SMC."""
    for experiment in results:
        experiment['time'] = load_runtime(experiment)
        experiment['tree_len'] = load_tree_len(experiment)
        experiment['n_iters'] = load_n_iters(experiment)
        if experiment['sampler'] == "smc":
            experiment['ess'] = load_ess(experiment)
    return results


def label_hierarchies(results):
    """Rewrite ss_prop as the block design the plots label runs by.

    HINTS: the hierarchy, e.g. ss_prop=0.125 -> [1, 2, 4, 8], meaning 1/ss_prop
    blocks at the bottom level, halving up to a single block at the top.
    Flat HINTS: the number of blocks, int(1 / ss_prop).
    The original proportion is kept as 'ss_frac', so calling this twice is safe."""
    for experiment in results:
        experiment.setdefault('ss_frac', experiment['ss_prop'])
        ss = experiment['ss_frac']
        if experiment['proposal_mechanism'] == "HINTS":
            experiment['ss_prop'] = [2 ** i for i in range(int(np.log2(round(1 / ss))) + 1)]
        elif experiment['proposal_mechanism'] == "Flat HINTS":
            experiment['ss_prop'] = int(round(1 / ss))
    return results


def _ss_frac(experiment):
    """The subsample proportion, before or after label_hierarchies; inf for MH."""
    ss = experiment.get('ss_frac', experiment['ss_prop'])
    return float('inf') if ss is None else ss


PROPOSAL_ORDER = ["DA", "Flat HINTS", "HINTS", "MH"]


def reorder_results(results):
    """Sort in place: DA, Flat HINTS, HINTS, MH, each by ss_prop ascending."""
    results.sort(key=lambda e: (PROPOSAL_ORDER.index(e['proposal_mechanism']), _ss_frac(e)))
    return results


def load_results(run_id, results_root="Results"):
    """(smc_results, mcmc_results) for one run directory, ready to plot:
    loaded, sorted with reorder_results, relabelled with label_hierarchies,
    and with the .h5 series attached."""
    smc, mcmc = load_metrics_files(run_id, results_root)
    for results in (smc, mcmc):
        reorder_results(results)
        label_hierarchies(results)
        attach_series(results)
    return smc, mcmc


def combine_results(results):
    """DA, Flat HINTS, HINTS, MH, each largest subsample first: the order
    plot_metric_split's legends and colours follow."""
    return [e for p in PROPOSAL_ORDER
            for e in sorted((e for e in results if e['proposal_mechanism'] == p),
                            key=_ss_frac, reverse=True)]


def by_proposal(results, order=("HINTS", "Flat HINTS", "DA", "MH")):
    """The experiments grouped in `order`, keeping their order within a group."""
    return [e for p in order for e in results if e['proposal_mechanism'] == p]


def mh_last(results):
    """Proposals alphabetically, MH last, each by its ss_prop label: the order
    the move outcome, fate and trace plots were drawn in."""
    def key(e):
        ss = e['ss_prop']
        return (e['proposal_mechanism'] == 'MH', e['proposal_mechanism'],
                float('inf') if ss is None else ss)
    return sorted(results, key=key)


# --------------------------------------------------------------------------- #
# labels shared by every plot
# --------------------------------------------------------------------------- #

def panel_title(experiment):
    """'Metropolis-Hastings', 'HINTS (Hierarchy=[1, 2, 4])' or 'DA (ss_prop=0.25)'."""
    p, ss = experiment['proposal_mechanism'], experiment['ss_prop']
    if p == "MH":
        return "Metropolis-Hastings"
    if "HINTS" in p:
        return f"{p} (Hierarchy={ss})"
    return f"{p} (ss_prop={ss})"


def series_label(experiment):
    """Legend entry for one experiment's line in plot_metric_split."""
    p, ss = experiment['proposal_mechanism'], experiment['ss_prop']
    if "HINTS" in p:
        return f"{p} (Hierarchy={ss})"
    if p == "DA":
        return f"DA (ss={ss})"
    return p


def _count_title(experiment, n):
    """Panel title carrying how many proposals the panel is made of."""
    p, ss = experiment['proposal_mechanism'], experiment['ss_prop']
    setting = f"Hierarchy={ss}" if "HINTS" in p else f"ss_prop={ss}"
    return f"{p} ({setting}, {n:,} proposals)"


def _panel_grid(n, ncols, panel_w, panel_h, **kwargs):
    """A grid of at least n panels, ncols wide, flattened into one array.
    Slots past the n-th are hidden."""
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_w, nrows * panel_h),
                             squeeze=False, **kwargs)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _figure_footnote(footnote):
    if footnote:
        plt.figtext(0.5, -0.05, footnote, wrap=True, horizontalalignment='center',
                    fontsize=10, fontstyle='italic')


# --------------------------------------------------------------------------- #
# metrics over iterations / time
# --------------------------------------------------------------------------- #

# Each subsample size keeps the same colour in every plot (see colour_index).
_LINES = ['#E69F00', '#56B4E9', '#CC79A7', '#F0E442', '#0072B2']
_SHADES = ['#F2BF99', '#BBE1F6', '#D9A3D9', '#FAF4B3', '#99C7E0']
LINE_COLOURS = {'MH': ['#009E73'], 'DA': _LINES, 'Flat HINTS': _LINES, 'HINTS': _LINES}
SHADE_COLOURS = {'MH': ['#99D8C7'], 'DA': _SHADES, 'Flat HINTS': _SHADES, 'HINTS': _SHADES}
X_LABELS = {"iteration": "Iteration", "time": "Sampling time (s)"}

# The panel of plot_metric_split each subsampled proposal is drawn in. MH is
# drawn in every panel as the reference.
SPLIT_PANELS = {"DA": 0, "Flat HINTS": 1, "HINTS": 2}


def colour_index(experiment, results):
    """Rank of this experiment's ss_prop among its proposal's ss_props, so each
    subsample size keeps the same colour in every plot."""
    def value(e):
        ss = e['ss_prop']
        return ss[-1] if isinstance(ss, list) else ss

    proposal = experiment['proposal_mechanism']
    ss_values = sorted({value(e) for e in results if e['proposal_mechanism'] == proposal},
                       key=lambda s: -1 if s is None else s)
    return ss_values.index(value(experiment)) % len(LINE_COLOURS[proposal])


def curve(experiment, key, x="iteration", n_grid=500):
    """(x values, mean, std) of a metric across runs, against iteration or,
    with x="time", against cumulative sampling time in seconds.

    `key` is looked up in experiment['data'] (the evaluated metrics) first and
    then on the experiment itself, where the .h5 series live ('tree_len', and
    'ess' for SMC)."""
    data = experiment['data']
    values = data[key] if key in data else experiment[key]   # (n_runs, n_points)
    if x == "iteration":
        return data['iterations'], values.mean(axis=0), values.std(axis=0)
    # Runs finish at different wall-clock times, so they are averaged on a
    # shared time grid that stops when the fastest run finished. At each grid
    # time a run holds the value of the last iteration it had completed.
    times = experiment['time']
    grid = np.linspace(0.0, times[:, -1].min(), n_grid)
    at_grid = np.stack([v[np.clip(np.searchsorted(t, grid, side="right") - 1, 0, None)]
                        for t, v in zip(times, values)])
    return grid, at_grid.mean(axis=0), at_grid.std(axis=0)


def plot_runtime_per_iter(results, samplername="SMC", ncols=2):
    """One panel per experiment: the time each iteration took, per chain and
    averaged over chains, on a log scale. Pass by_proposal(...) for the
    notebooks' panel order."""
    fig, axes = _panel_grid(len(results), ncols, 6, 4, sharey=True)
    for ax, experiment in zip(axes, results):
        per_iter = np.diff(experiment['time'], axis=1)          # (n_runs, n_points - 1)
        for j, chain in enumerate(per_iter):
            ax.plot(chain, linewidth=0.75, color="tab:blue", alpha=0.2,
                    label="chain runtime per iter" if j == 0 else None)
        ax.plot(per_iter.mean(axis=0), linewidth=1.25, color="blue", alpha=1, label="mean runtime")
        ax.set_title(panel_title(experiment))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Run Time (s)')
        ax.set_yscale('log')
    plt.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Run Time Per Iteration for {samplername} Experiments - {results[0]['run_id']}",
                 fontsize=16, y=1.035)
    plt.show()


def plot_ess(results, ncols=2, clip_to=None):
    """One panel per SMC experiment: mean ESS across runs with a +/- 1 std
    band (clipped to [0, N]) and the N/2 resampling threshold.

    clip_to="HINTS" limits every panel's x axis to the longest HINTS run, so
    runs that went on longer (MH, DA) are compared over the same steps."""
    limit = None
    if clip_to is not None:
        limit = 1 + max(e['data']['iterations'][-1] for e in results
                        if e['proposal_mechanism'] == clip_to)
    fig, axes = _panel_grid(len(results), ncols, 6, 4, sharey=True, sharex=limit is None)
    for ax, experiment in zip(axes, results):
        iters = experiment['data']['iterations']
        mean, std = experiment['ess'].mean(axis=0), experiment['ess'].std(axis=0)
        n = experiment['particles']
        ax.plot(iters, mean, label='Mean ESS', linewidth=1)
        ax.fill_between(iters, np.maximum(0, mean - std), np.minimum(n or np.inf, mean + std),
                        alpha=0.3, label='Std Dev')
        if n:
            ax.axhline(y=n // 2, color='red', linestyle='--', label='N/2', linewidth=2)
        ax.set_title(panel_title(experiment))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Effective Sample Size (ESS)')
        if limit is not None:
            ax.set_xlim([-1, limit])
    plt.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.02))
    title = f"Effective Sample Size (ESS) for SMC Experiments - {results[0]['run_id']}"
    if limit is not None:
        title += f" for {limit} iterations"
    fig.suptitle(title, fontsize=16, y=1.035)
    plt.show()


def plot_metric_split(results, key, title, samplername, ylabel, cols=3, x="iteration", footnote=None,
                      xlim=None, yscale=None, sharex=True, sharey=True, ignore_mh=False):
    """One panel each for DA, Flat HINTS and HINTS: the mean of `key` across
    runs with a +/- 1 std band, one line per subsample size, and MH drawn in
    every panel as the reference (unless ignore_mh). `x` is "iteration" or
    "time". Pass combine_results(...) for the notebooks' legend order."""
    fig, axes = plt.subplots(1, cols, figsize=(20, 6), sharey=sharey, sharex=sharex, squeeze=False)
    axes = axes.ravel()
    for experiment in results:
        proposal = experiment['proposal_mechanism']
        if proposal == "MH":
            if ignore_mh:
                continue
            targets = axes
        else:
            targets = [axes[SPLIT_PANELS[proposal]]]
        k = colour_index(experiment, results)
        xs, mean, std = curve(experiment, key, x)
        lo, hi = mean - std, mean + std
        if key == "tree_len":
            cap = experiment['max_tree_size']
            lo, hi = np.maximum(0, lo), np.minimum(np.inf if cap is None else cap, hi)
        for ax in targets:
            ax.fill_between(xs, lo, hi, color=SHADE_COLOURS[proposal][k], alpha=0.5, zorder=1)
            ax.plot(xs, mean, label=series_label(experiment), color=LINE_COLOURS[proposal][k], zorder=2)

    particles = results[0]['particles']
    for proposal, i in SPLIT_PANELS.items():
        ax = axes[i]
        if key == "ess" and particles:
            ax.axhline(y=particles // 2, color='red', linestyle='--', label='N/2', linewidth=2)
        ax.set_title(proposal)
        ax.set_xlabel(X_LABELS[x])
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if yscale:
            ax.set_yscale(yscale)
        ax.legend()

    first = results[0]
    span = f" (First {max(xlim)} {'seconds' if x == 'time' else 'iterations'})" if xlim else ""
    fig.suptitle(f"{samplername} {title} by {x} on {first['dataset']} "
                 f"for {first['data']['n_runs']} runs{span}")
    _figure_footnote(footnote)
    plt.show()


def mh_footnote(results, unit):
    """How MH's run length differs from the rest, read off the files themselves.
    `unit` is "step" (SMC) or "iteration" (MCMC). None if there is no MH run."""
    mh = [e for e in results if e['proposal_mechanism'] == 'MH']
    rest = ([e for e in results if e['proposal_mechanism'] == 'HINTS']
            or [e for e in results if e['proposal_mechanism'] != 'MH'])
    if not mh or not rest:
        return None
    mh, rest = mh[0], rest[0]
    note = (f"MH and DA were run for {mh['n_iters'] / rest['n_iters']:g} times as many "
            f"{unit}s as the other proposal mechanisms")
    if mh['stride'] > 1:
        note += (f", with its metrics evaluated every {mh['stride']}nd {unit} "
                 f"and plotted at those {unit}s")
    return note + "."


def plot_chains(results, key, samplername, label, x="iteration", footnote=None, xlim=None, ncols=None):
    """One panel per experiment: every chain of `key`, with the mean and
    +/- 1 std across chains. `x` is "iteration" or "time". The grid is as many
    columns wide as there are HINTS runs unless ncols is given."""
    if ncols is None:
        ncols = max(1, sum(e['proposal_mechanism'] == "HINTS" for e in results))
    fig, axes = _panel_grid(len(results), ncols, 6, 4, sharey=True)
    for ax, experiment in zip(axes, results):
        data = experiment['data']
        values = data[key] if key in data else experiment[key]
        if x == "iteration":
            xs = data['iterations']
            chain_xs = [xs] * len(values)
        else:
            chain_xs = experiment['time']
            xs = chain_xs.mean(axis=0)
        mean, std = values.mean(axis=0), values.std(axis=0)
        for j, (cx, chain) in enumerate(zip(chain_xs, values)):
            ax.plot(cx, chain, linewidth=1, color="blue", alpha=0.25, label="chain" if j == 0 else None)
        ax.plot(xs, mean, linewidth=2, color="red", alpha=1, label="mean")
        ax.plot(xs, mean + std, linewidth=2, color="orange", linestyle="--", alpha=1, label="+1 std")
        ax.plot(xs, mean - std, linewidth=2, color="orange", linestyle="--", alpha=1, label="-1 std")
        ax.set_title(panel_title(experiment))
        ax.set_xlabel("Iterations" if x == "iteration" else "Time (s)")
        ax.set_ylabel(label)
        if xlim is not None:
            ax.set_xlim(xlim)
    plt.tight_layout()

    legend = {}                              # one entry per label, across panels
    for ax in axes:
        for handle, name in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(name, handle)
    fig.legend(list(legend.values()), list(legend), loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))

    per, unit = ("iteration", "iterations") if x == "iteration" else ("second", "seconds")
    span = f" ({xlim[0]}-{xlim[1]} {unit})" if xlim is not None else ""
    fig.suptitle(f"{label} Per {per} for {samplername} Experiments{span} - {results[0]['run_id']}",
                 fontsize=16, y=1.035)
    _figure_footnote(footnote)
    plt.show()


# --------------------------------------------------------------------------- #
# move outcomes and fates, by move type
# --------------------------------------------------------------------------- #
#
# Every Run_N in an .h5 holds the move log: "move" (an index into MOVES) and
# "outcome" (an index into OUTCOMES) per move, and "call", the sample() call
# each move came from.

# Segments stack in outcome order, so the colours were chosen to pass
# colour-blindness and contrast checks between neighbouring segments in that
# order.
OUTCOME_COLOURS = {
    STAY:         "#8c8578",  # warm grey: the null outcome
    BARRED:       "#e27860",  # orange-red
    SCREENED_OUT: "#a59ae6",  # violet
    INADMISSIBLE: "#e9de65",  # amber
    PROPOSED:     "#8bb4e5",  # blue
    APPLIED:      "#6AE26A",  # green
}

SURFACE = "#ffffff"
BAR_W = 0.5
INSIDE_MIN = 0.055   # segments thinner than this fraction of the axis get a lifted label
GAP = 0.052          # vertical spacing of the lifted labels, same units

# plot_diagnostics.FATES in order, under the names the plots use. The last two
# never reached the target, but they are most of the work on some move types,
# so dropping them would rescale the bars to only part of what the proposal did.
FATE_LABELS = ("Approx Accept + Accept", "MH Accept", "Approx Accept + Reject",
               "Approx Reject + Accept", "Approx Reject", "MH Reject",
               "barred / inadmissible", "no move available")

FATE_COLOURS = {
    0: "#98df8a",  # light green: passed the screen and the sampler took it
    1: "#22A822",  # dark green: accepted with no screen (MH, DA's skip path)
    2: "#ff7f0e",  # orange: the screen passed it, the sampler did not
    3: "#54c8e2",  # cyan: HINTS, the sweep landed and this move did not
    4: "#ff9896",  # pink: the surrogate threw it out
    5: "#e34948",  # red: rejected, never screened
    6: "#8c8578",  # warm grey
    7: "#c7c2b8",  # light grey
}

# The coarse view of what became of a move. "Accept" is plot_diagnostics' own
# acceptance numerator, so a move screened out of a HINTS sweep that landed
# counts as a Reject: the sweep was accepted, that move was not.
FATE_GROUPS = (
    ("Accept",      "#22A822", ACCEPTED_FATES),
    ("Reject",      "#e34948", tuple(k for k in range(6) if k not in ACCEPTED_FATES)),
    ("Barred",      "#8c8578", (6,)),
    ("unavailable", "#c7c2b8", (7,)),
)

# Footnotes the notebooks put under the SMC fate plots.
SMC_FATE_NOTE = ("SMC has no accept/reject step -- it reweights, so nothing "
                 "lands in the Reject fates and Accept means the proposal survived.")
SMC_FATE_NOTE_GROUPED = ("SMC has no accept/reject step -- it reweights, so nothing "
                         "lands in Reject and Accept means the proposal survived.")


def move_outcome_counts(experiment):
    """(len(MOVES), len(OUTCOMES)) counts, pooled over every chain of a run."""
    with h5py.File(experiment['h5_path'], "r") as f:
        runs = _run_names(f)
        moves = np.concatenate([f[run]["move"][()] for run in runs])
        outcomes = np.concatenate([f[run]["outcome"][()] for run in runs])
    flat = moves.astype(np.int64) * len(OUTCOMES) + outcomes.astype(np.int64)
    return np.bincount(flat, minlength=len(MOVES) * len(OUTCOMES)).reshape(len(MOVES), len(OUTCOMES))


def move_fate_counts(experiment):
    """(len(MOVES), len(FATE_LABELS)) counts, pooled over every chain. Each
    run's moves are classified by plot_diagnostics.move_fates."""
    with h5py.File(experiment['h5_path'], "r") as f:
        tables = [move_fates(f[run]) for run in _run_names(f)]
    return sum((t for t in tables if t is not None),
               np.zeros((len(MOVES), len(FATE_LABELS)), dtype=np.int64))


def group_fates(count):
    """Collapse a (len(MOVES), len(FATE_LABELS)) table onto FATE_GROUPS."""
    return np.stack([count[:, list(cols)].sum(axis=1) for _, _, cols in FATE_GROUPS], axis=1)


def _si(n):
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= cut:
            return f"{n / cut:.1f}{suffix}".replace(".0", "")
    return str(int(n))


def _pct(value):
    return f"{value:.2f}%" if value >= 1 else f"{value:.3f}%"


def _n_lifted(heights, span):
    """How many segments are too thin to hold their own label."""
    return sum(1 for h in heights if 0 < h < INSIDE_MIN * span)


def _label_bar(ax, i, heights, shares, span, colours, inside_size=10, lifted_size=10, stroke_width=1.5):
    """Percentages beside the segments they belong to, each in its segment's
    colour with a black outline. The outline keeps a label readable on its
    own fill or on the white background above the bar. Segments too thin
    for a label get one lifted above the bar, joined by a leader line."""
    # A heavy outline swallows the text at this size, so keep it thin.
    stroke = [pe.withStroke(linewidth=stroke_width, foreground="black")]
    gap = GAP * span
    y = heights.sum() + gap
    j = 0
    for k in range(len(heights)):
        if heights[k] == 0:
            continue
        centre = heights[:k].sum() + heights[k] / 2
        if heights[k] >= INSIDE_MIN * span:
            ax.text(i, centre, _pct(shares[k]), ha='center', va='center',
                    color=colours[k], size=inside_size, weight='bold',
                    zorder=4, path_effects=stroke)
        else:
            # Out of the segment sideways, then straight up to the label, so a
            # leader never crosses the stack or a neighbouring bar.
            ax.annotate(_pct(shares[k]), xy=(i + BAR_W / 2, centre),
                        xytext=(i + BAR_W / 2 + 0.05 + 0.055 * j, y),
                        ha='left', va='center', size=lifted_size, weight='bold', zorder=4,
                        color=colours[k], path_effects=stroke,
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', lw=0.8, shrinkA=0, shrinkB=2,
                                        color=colours[k],
                                        connectionstyle="angle,angleA=0,angleB=90"))
            y += gap
            j += 1


def _move_panels(results, counts, colours, share, ncols, tick_prefix, label_style):
    """The panels shared by plot_move_outcomes and plot_move_fates: one per
    experiment, a stacked bar per move type. Returns (fig, axes)."""
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.4, nrows * 4.4), squeeze=False)
    axes = axes.ravel()
    xs = np.arange(len(MOVES))
    n_stack = counts[0].shape[1]

    panels = []
    for count in counts:
        totals = count.sum(axis=1)
        shares = np.divide(count * 100.0, totals[:, None], out=np.zeros(count.shape),
                           where=totals[:, None] > 0)
        heights = shares if share else count.astype(float)
        span = 100.0 if share else max(heights.sum(axis=1).max(), 1.0)
        # Headroom for the tallest column of lifted labels, so none is drawn
        # outside the axes (tight_layout would squash the panel to fit it).
        n_lift = max(_n_lifted(heights[i], span) for i in range(len(MOVES)))
        panels.append((totals, shares, heights, span, span * (1 + GAP * (n_lift + 0.8))))

    # A shared limit only makes sense when every panel is on the same 0-100 scale.
    shared_top = max(p[-1] for p in panels) if share else None

    for ax, experiment, (totals, shares, heights, span, top) in zip(axes, results, panels):
        bottom = np.zeros(len(MOVES))
        for k in range(n_stack):
            ax.bar(xs, heights[:, k], width=BAR_W, bottom=bottom, zorder=2,
                   color=colours[k], edgecolor=SURFACE, linewidth=1.5)
            bottom += heights[:, k]
        for i in range(len(MOVES)):
            if totals[i] > 0:
                _label_bar(ax, i, heights[i], shares[i], span, colours, **label_style)

        ax.set_title(_count_title(experiment, totals.sum()), fontsize=14, fontweight='bold')
        # Under each move: how many proposals it drew, and what share of all
        # the run's proposals that is (the stack above is a share of n alone).
        move_share = 100.0 * totals / max(totals.sum(), 1)
        ax.set_xticks(xs, [f"{m}\n{tick_prefix}{_si(t)}\n{_pct(s)}"
                           for m, t, s in zip(MOVES, totals, move_share)])
        ax.set_xlim(-0.55, len(MOVES) - 0.15)
        ax.set_ylim(0, shared_top if share else top)
        ax.set_xlabel('Move type')
        ax.set_ylabel('Share of proposals (%)' if share else 'Proposals')
        ax.grid(axis='x', visible=False)
        if share:
            ax.set_yticks(np.arange(0, 101, 25))

    for ax in axes[len(results):]:
        ax.set_visible(False)
    return fig, axes


def plot_move_outcomes(results, samplername, share=True, footnote=None, ncols=3):
    """One panel per experiment: what the proposal did with each move
    (OUTCOMES), stacked by move type.

    share=True stacks every bar to 100% and prints its total underneath. The
    totals differ by orders of magnitude between moves and experiments, so
    raw counts would flatten all but the largest bar. share=False plots the
    counts, each panel on its own y axis."""
    counts = [move_outcome_counts(e) for e in results]
    fig, _ = _move_panels(results, counts, OUTCOME_COLOURS, share, ncols, "n=", {})
    fig.legend(handles=[Patch(facecolor=OUTCOME_COLOURS[k], label=OUTCOMES[k]) for k in range(len(OUTCOMES))],
               loc='upper center', ncol=len(OUTCOMES), frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"Proposal outcome by move type -- {samplername} experiments, {results[0]['run_id']}",
                 fontsize=16)
    if footnote:
        fig.text(0.5, -0.01, footnote, ha='center', fontsize=10, fontstyle='italic')
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    plt.show()


def plot_move_fates(results, samplername, share=True, footnote=None, ncols=3, grouped=False):
    """One panel per experiment: where each move type's proposals ended up,
    with the subsample screen crossed with the sampler's own accept/reject
    (FATE_LABELS).

    grouped=True collapses the fates onto Accept / Reject, keeping barred and
    unavailable separate. Neither reached the outer step, so counting them as
    Reject would suggest the sampler turned them down."""
    counts = [move_fate_counts(e) for e in results]
    if grouped:
        counts = [group_fates(count) for count in counts]
        labels = [name for name, _, _ in FATE_GROUPS]
        colours = {k: hexcode for k, (_, hexcode, _) in enumerate(FATE_GROUPS)}
    else:
        labels, colours = list(FATE_LABELS), FATE_COLOURS
    fig, axes = _move_panels(results, counts, colours, share, ncols, "",
                             dict(inside_size=10.5, lifted_size=9, stroke_width=1.4))
    for ax in axes:                          # y labels on the first column only
        if ax.get_subplotspec().colspan.start > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    fig.legend(handles=[Patch(facecolor=colours[k], label=labels[k]) for k in range(len(labels))],
               loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    first = results[0]
    fig.suptitle(f"Move Outcomes over {first['data']['n_runs']} chains in {samplername} "
                 f"experiments on {first['dataset']}", fontsize=20, y=1, fontweight='bold')
    if footnote:
        fig.text(0.5, -0.01, footnote, ha='center', fontsize=10, fontstyle='italic')
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    plt.show()


def move_fate_table(results, samplername=None, grouped=False):
    """One dict per experiment, keyed '<move>_<fate>': the raw count of that
    fate with its share of the move's proposals underneath, as LaTeX
    \\makecell entries ready for create_latex_table. Empty cells are left out."""
    counts = [move_fate_counts(e) for e in results]
    if grouped:
        counts = [group_fates(count) for count in counts]
        labels = [name for name, _, _ in FATE_GROUPS]
    else:
        labels = list(FATE_LABELS)

    table = []
    for experiment, count in zip(results, counts):
        totals = count.sum(axis=1)
        shares = np.divide(count, totals[:, None], out=np.zeros(count.shape), where=totals[:, None] > 0)
        p, ss = experiment['proposal_mechanism'], experiment['ss_prop']
        if "HINTS" in p:
            name = f"\\makecell{{{p} \\\\ ${ss}$}}"
        elif p == "DA":
            name = f"\\makecell{{{p} \\\\ {float(ss):.3g}}}"
        else:
            name = "\\makecell{MH}"
        row = {"proposal_mechanism": name}
        for i, move in enumerate(MOVES):
            for j, fate in enumerate(labels):
                if count[i, j] > 0:
                    row[f"{move}_{fate}"] = f"\\makecell{{{count[i, j]:,} \\\\ ({shares[i, j]:.2g})}}"
        table.append(row)
    return table


def create_latex_table(counts, samplername=None):
    """A LaTeX table of move_fate_table's rows: each move a column heading with
    its fates as subheadings, and the highest share in every column in bold."""
    df = pd.DataFrame(counts).fillna("--")
    df.columns = pd.MultiIndex.from_tuples(
        [(col, "") if col == "proposal_mechanism" else tuple(col.split("_", 1)) for col in df.columns])
    for col in df.columns:
        if col[0] not in MOVES:
            continue
        shares = df[col].str.extract(r"\(([\d\.e\-]+)\)")[0].astype(float)
        if shares.notna().any():
            best = shares.idxmax()
            df.loc[best, col] = "\\textbf{" + df.loc[best, col] + "}"
    return df.to_latex(index=False, escape=False, multicolumn=True, multicolumn_format='c',
                       column_format='l' + 'c' * (len(df.columns) - 1), multirow=False)


# --------------------------------------------------------------------------- #
# acceptance by move type, below vs at the tree-size cap
# --------------------------------------------------------------------------- #
#
# Each move is split on whether the tree it was proposed on was already at
# max_tree_size. At the cap a grow is barred outright, so prune and change are
# the only ways to restructure the tree there.
#
# The tree a move was proposed on is the state *before* its sample() call:
#   MCMC: the previous iteration's state (the initial tree for iteration 0).
#   SMC:  the particle's ancestor after resampling, which the .h5 does not
#         record. Systematic resampling hands out copies in ancestor order, one
#         assignment per interval of its uniform u, so the candidates are
#         enumerated from the stored weights and kept only if they agree with
#         what is known: a particle whose call stayed holds its ancestor's
#         slot, and one that moved changed size by at most the grows and
#         prunes its call applied. A call whose remaining candidates disagree
#         on whether the ancestor was at the cap is left out and counted.
# For a HINTS sweep, "at the cap" means the sweep started there. A prune early
# in the sweep can free room for a grow later in it. That grow was drawn on a
# tree below the cap (select_move gives grow probability 0 at the cap, judged
# on the sweep's working tree), so every grow is counted as below the cap.
# MCMC rates are the outer accept. SMC never rejects, so its rate means
# "kept": the move survived the proposal (see move_fates).

_GROW, _PRUNE = MOVE_CODE["grow"], MOVE_CODE["prune"]
_TOOK = (PROPOSED, APPLIED)

CAP_MOVES = ("grow", "prune", "change")      # stay has no fate worth splitting
CAP_BAR_W = 0.36                             # two bars per move, side by side
CAP_GROUPS = FATE_GROUPS[:3]                 # Accept, Reject, Barred

CAP_NOTE = ("Each move is placed by the tree its sample() call started on; a grow is never "
            "drawn on a full tree, so a HINTS sweep's grow after an in-sweep prune counts as below cap.")
SMC_CAP_NOTE = ("SMC has no accept/reject step -- it reweights, so Reject here is only the "
                "subsample screen (DA, HINTS) and Accept means the proposal survived.\n" + CAP_NOTE)


def _systematic_ancestors(logw):
    """Every ancestor vector systematic_resampling can draw from these
    weights: one per interval of u that changes the copy counts."""
    w = np.exp(logw - logw.max())
    w = (w / w.sum()).astype(np.float32)
    N = len(w)
    cdf = np.cumsum(w * N)
    prev = cdf - w * N
    breaks = np.unique(np.r_[0.0, np.mod(cdf, 1.0), 1.0])
    out = []
    for u in (breaks[:-1] + breaks[1:]) / 2:
        ncopies = (np.ceil(cdf - u) - np.ceil(prev - u)).astype(int)
        excess = ncopies.sum() - N
        if excess:   # check_stability's fix-up
            ncopies[np.nonzero(ncopies > 0)[0][-1]] -= excess
        if (ncopies >= 0).all():
            out.append(np.repeat(np.arange(N), ncopies))
    return out


def pre_call_at_cap(run, init_size, cap):
    """Per sample() call: 1 if the tree it was handed was at the cap, 0 if
    not, -1 if that can't be determined (SMC, see above, or the first call
    of a run whose start was drawn from the prior: init_size None, as the
    starting trees are not stored)."""
    slots = run["state_index"][()]
    sizes = run["state_tree_lengths"][()][slots]               # (T, N)
    T, N = slots.shape
    if "resampled" not in run:                                 # MCMC, N == 1
        pre = (np.r_[init_size or 0, sizes[:-1, 0]] >= cap).astype(np.int8)
        if init_size is None:
            pre[0] = -1
        return pre

    resampled = run["resampled"][()]
    logw = run["state_log_weights"][()].astype(np.float64)
    took = (run["call_outcome"][()] == PROPOSED).reshape(T, N)
    call, move, outcome = run["call"][()], run["move"][()], run["outcome"][()]
    ok = np.isin(outcome, _TOOK)
    grows = np.bincount(call[ok & (move == _GROW)], minlength=T * N).reshape(T, N)
    prunes = np.bincount(call[ok & (move == _PRUNE)], minlength=T * N).reshape(T, N)
    lo = np.where(took, sizes - grows, sizes)
    hi = np.where(took, sizes + prunes, sizes)

    at_cap = np.full((T, N), -1, dtype=np.int8)
    if init_size is not None:
        at_cap[0] = init_size >= cap                           # all start on the init tree
    for t in range(1, T):
        prev_cap = sizes[t - 1] >= cap
        if not resampled[t]:
            at_cap[t] = prev_cap
            continue
        if prev_cap.all() or not prev_cap.any():               # every ancestor agrees
            at_cap[t] = prev_cap[0]
            continue
        stay = ~took[t]
        fits = [anc for anc in _systematic_ancestors(logw[t - 1])
                if (slots[t - 1, anc][stay] == slots[t][stay]).all()
                and ((sizes[t - 1, anc] >= lo[t]) & (sizes[t - 1, anc] <= hi[t])).all()]
        if not fits:
            continue
        flags = np.stack([prev_cap[anc] for anc in fits])
        agree = (flags == flags[0]).all(axis=0)
        at_cap[t, agree] = flags[0, agree]
    return at_cap.ravel()


def _init_size(experiment):
    """Internal nodes in the tree the runs started from; None when each run
    drew its own from the prior."""
    if experiment.get("init_from_prior"):
        return None
    if experiment.get("init_tree") is not None:
        return len(experiment["init_tree"])
    return experiment.get("init_tree_size") or 0


def move_fate_counts_by_cap(experiment):
    """({True: at-cap table, False: below-cap table}, unknown, total): two
    tables shaped like move_fate_counts', pooled over every chain, and how
    many of the total moves were left out because their side of the cap
    couldn't be determined."""
    cap = experiment["max_tree_size"]
    init_size = _init_size(experiment)
    empty = np.zeros((len(MOVES), len(FATE_LABELS)), dtype=np.int64)
    tables = {True: empty, False: empty}
    unknown = total = 0
    with h5py.File(experiment["h5_path"], "r") as f:
        for name in _run_names(f):
            run = f[name]
            call = run["call"][()]
            per_move = pre_call_at_cap(run, init_size, cap)[call]
            per_move[run["move"][()] == _GROW] = 0     # never drawn on a full tree
            calls = {k: run[k][()] for k in ("call_id", "call_outcome", "accepted") if k in run}
            moves = {k: run[k][()] for k in ("move", "outcome", "dsurr")}
            moves["call"] = call
            for flag in (True, False):
                keep = per_move == flag
                sub = dict(calls, **{k: v[keep] for k, v in moves.items()})
                tables[flag] = tables[flag] + move_fates(sub)
            unknown += int((per_move < 0).sum())
            total += len(per_move)
    return tables, unknown, total


def _exp_label(experiment):
    p = experiment["proposal_mechanism"]
    return "MH" if p == "MH" else f"{p} {experiment['ss_prop']}"


def cap_acceptance_table(results):
    """(DataFrame, counts). The DataFrame has one row per experiment: for each
    move type, the accept (MCMC) or kept (SMC) rate below the cap and at it,
    with how many moves that rate is out of. counts is
    [(experiment, {True: at-cap, False: below-cap table})] for plot_cap_fates
    and plot_cap_fates_by_move, so the .h5 files are only read once."""
    rows, counts = [], []
    for experiment in results:
        tables, unknown, total = move_fate_counts_by_cap(experiment)
        counts.append((experiment, tables))
        row = {"experiment": _exp_label(experiment)}
        for flag, where in ((False, "below cap"), (True, "AT CAP")):
            grouped = group_fates(tables[flag])                # Accept, Reject, Barred, unavailable
            for m in CAP_MOVES:
                i = MOVE_CODE[m]
                n = grouped[i].sum()
                row[(m, where)] = f"{grouped[i, 0] / n:.2%} ({n:,})" if n else "--"
        row[("", "unknown")] = f"{unknown / total:.2%}" if total else "--"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("experiment")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df, counts


def _cap_shares(tables, flag, move):
    """(shares in %, n) of Accept / Reject / Barred for one move on one side
    of the cap; (None, 0) if no such move was proposed."""
    row = group_fates(tables[flag])[MOVE_CODE[move], :3]
    n = row.sum()
    return (100.0 * row / n if n else None), n


def _cap_headroom(counts):
    """Top of the y axis, with room for the tallest stack of above-bar labels
    in any panel. Accept is labelled at the foot of its bar, so it never needs any."""
    n_above = max((sum(1 for s in shares[1:] if 0 < s < INSIDE_MIN * 100)
                   for _, tables in counts for flag in (False, True) for m in CAP_MOVES
                   for shares, n in [_cap_shares(tables, flag, m)] if n),
                  default=0)
    return 100.0 * (1 + GAP * (n_above + 0.6))


def _label_cap_bar(ax, x, shares, colours):
    """Each share on its own segment, or, if the segment is too thin, stacked
    above the bar in the segment's colour. Labels go straight above rather
    than to the side, so the two bars of a pair never cross. Exception: a thin
    Accept segment (k == 0) is labelled at the foot of the bar, next to the
    baseline it is measured from."""
    stroke = [pe.withStroke(linewidth=1.4, foreground="black")]
    y = 100.0 + GAP * 100.0 * 0.6
    for k, share in enumerate(shares):
        if share == 0:
            continue
        if k == 0 and share < INSIDE_MIN * 100.0:
            ax.text(x, 1.0, _pct(share), ha='center', va='bottom',
                    color=colours[k], size=10, weight='bold', zorder=4,
                    path_effects=stroke)
        elif share >= INSIDE_MIN * 100.0:
            ax.text(x, shares[:k].sum() + share / 2, _pct(share), ha='center',
                    va='center', color=colours[k], size=10, weight='bold',
                    zorder=4, path_effects=stroke)
        else:
            ax.text(x, y, _pct(share), ha='center', va='center', color=colours[k],
                    size=8.5, weight='bold', zorder=4, path_effects=stroke)
            y += GAP * 100.0


def _cap_bar(ax, x, shares, width, colours, linewidth):
    bottom = 0.0
    for k in range(len(CAP_GROUPS)):
        ax.bar(x, shares[k], width=width, bottom=bottom, zorder=2,
               color=colours[k], edgecolor=SURFACE, linewidth=linewidth)
        bottom += shares[k]
    _label_cap_bar(ax, x, shares, colours)


def _cap_legend(fig, colours, y):
    fig.legend(handles=[Patch(facecolor=colours[k], label=name) for k, (name, _, _) in enumerate(CAP_GROUPS)],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


def plot_cap_fates(counts, samplername, ncols=3, footnote=None):
    """One panel per experiment, in plot_move_fates' style: for each move
    type, a pair of 100% bars (below the cap, at it) stacked by Accept /
    Reject / Barred. A grow is never drawn at the cap, so that slot says so.
    `counts` is the second value cap_acceptance_table returns."""
    colours = {k: hexcode for k, (_, hexcode, _) in enumerate(CAP_GROUPS)}
    fig, axes = _panel_grid(len(counts), ncols, 6, 4.8)
    centres = np.arange(len(CAP_MOVES))
    offsets = {False: -CAP_BAR_W / 2 - 0.035, True: CAP_BAR_W / 2 + 0.035}
    top = _cap_headroom(counts)

    for ax, (experiment, tables) in zip(axes, counts):
        ticks, labels, n_props = [], [], 0
        for flag in (False, True):
            for c, m in zip(centres, CAP_MOVES):
                x = c + offsets[flag]
                shares, n = _cap_shares(tables, flag, m)
                n_props += n
                ticks.append(x)
                labels.append(f"{'AT CAP' if flag else 'BELOW CAP'}\n{_si(n) if n else '–'}")
                if not n:
                    ax.bar(x, 100, width=CAP_BAR_W, color="none", edgecolor="#c7c2b8",
                           hatch="//", linewidth=0.8, zorder=1)
                    ax.text(x, 50, "never\nproposed", ha="center", va="center",
                            size=9, color="#8c8578", style="italic", zorder=4)
                    continue
                _cap_bar(ax, x, shares, CAP_BAR_W, colours, 1.5)

        order = np.argsort(ticks)
        ax.set_xticks(np.array(ticks)[order], np.array(labels)[order], fontsize=9)
        # The move name once per pair, under its two bars' own labels.
        for c, m in zip(centres, CAP_MOVES):
            ax.annotate(m, xy=(c, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -38), textcoords="offset points",
                        ha="center", va="top", fontsize=11, fontweight="bold")
        ax.set_title(_count_title(experiment, n_props), fontsize=14, fontweight='bold')
        ax.set_xlim(-0.6, len(CAP_MOVES) - 0.4)
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, 101, 25))
        ax.set_ylabel('Share of proposals (%)')
        ax.grid(axis='x', visible=False)

    for ax in axes:                          # y labels on the first column only
        if ax.get_subplotspec().colspan.start > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    _cap_legend(fig, colours, 0.985)
    experiment = counts[0][0]
    fig.suptitle(f"Move Outcomes Below vs At max_tree_size over {experiment['data']['n_runs']} chains "
                 f"in {samplername} experiments on {experiment['dataset']}",
                 fontsize=20, y=1, fontweight='bold')
    if footnote:
        fig.text(0.5, -0.01, footnote, ha='center', fontsize=10, fontstyle='italic')
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), h_pad=2.5)
    plt.show()


def _wrapped_label(experiment):
    """A tick label with one word per line, and the subsample setting (ss_prop,
    or the HINTS hierarchy without spaces so it fits under one bar) on its own
    line beneath the proposal's name."""
    p = experiment["proposal_mechanism"]
    lines = p.split()
    if p != "MH":
        lines.append(str(experiment["ss_prop"]).replace(" ", ""))
    return "\n".join(lines)


def plot_cap_fates_by_move(counts, samplername, footnote=None):
    """plot_cap_fates' counts turned around: one panel for moves proposed
    below max_tree_size and one for moves proposed at it, with every
    experiment side by side within each move type. This compares algorithms
    directly, where plot_cap_fates compares below and at the cap within one."""
    colours = {k: hexcode for k, (_, hexcode, _) in enumerate(CAP_GROUPS)}
    n_exp = len(counts)
    pitch = 0.9 / n_exp                          # bar spacing within a move's group
    width = pitch * 0.85
    top = _cap_headroom(counts)

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    for ax, (flag, where) in zip(axes, ((False, "Moves Proposed Below Max Tree Size Cap"),
                                        (True, "Moves Proposed at Max Tree Size Cap"))):
        ticks, labels = [], []
        for c, m in enumerate(CAP_MOVES):
            drawn = False
            for e, (experiment, tables) in enumerate(counts):
                x = c + (e - (n_exp - 1) / 2) * pitch
                ticks.append(x)
                labels.append(_wrapped_label(experiment))
                shares, n = _cap_shares(tables, flag, m)
                if not n:
                    continue
                drawn = True
                _cap_bar(ax, x, shares, width, colours, 1.2)
            if not drawn:
                ax.text(c, 50, f"{m.capitalize()} is never proposed\non a tree at the cap",
                        ha='center', va='center', size=13, style='italic', color="#8c8578")
            # The move's name over its group, and a rule between groups.
            ax.annotate(m.capitalize(), xy=(c, 1), xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', size=15, weight='bold', style='italic')
            if c:
                ax.axvline(c - 0.5, color="#c7c2b8", linewidth=1.2, zorder=1)

        ax.set_xticks(ticks, labels, fontsize=10, weight='bold', style='italic')
        ax.set_xlim(-0.5, len(CAP_MOVES) - 0.5)
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, 101, 25))
        ax.set_ylabel("Share of proposals (%)", fontsize=12)
        # The row's heading, lifted clear of the move names above the groups.
        ax.set_title(where, fontsize=17, fontweight='bold', pad=30)
        ax.grid(axis='x', visible=False)

    _cap_legend(fig, colours, 0.975)
    experiment = counts[0][0]
    fig.suptitle(f"Move Outcomes by Move Type, Below vs At max_tree_size, over "
                 f"{experiment['data']['n_runs']} chains in {samplername} experiments "
                 f"on {experiment['dataset']}", fontsize=20, y=1, fontweight='bold')
    if footnote:
        fig.text(0.5, -0.01, footnote, ha='center', fontsize=10, fontstyle='italic')
    fig.tight_layout(rect=(0, 0.02, 1, 0.965), h_pad=2.5)
    plt.show()


# --------------------------------------------------------------------------- #
# proposal outcome traces
# --------------------------------------------------------------------------- #
#
# What each proposal came to, over the run: at every iteration (MCMC) or step
# (SMC), the share of chains or particles in each state.
#
#   Accept                           the move (or the HINTS sweep) became the
#                                    new state
#   Reject                           an unscreened proposal turned down by the
#                                    outer MH step (MH, and DA's skip path)
#   Stage 1 accept + stage 2 reject  passed the subsample screen (DA) or every
#                                    level of the sweep (HINTS), then the outer
#                                    MH step on the full data turned it down
#   Stage 1 reject                   the screen threw it out: the state is
#                                    handed straight back and the full target
#                                    is never evaluated
#   No valid move                    barred, inadmissible, or nothing to draw:
#                                    the state is handed back without a screen
#
# A (Flat) HINTS call is a sweep of many moves, one per block, so a second
# figure follows the moves inside the sweeps, in the same colours:
#
#   Accept                           applied by its block, and the sweep landed
#   Stage 1 accept + stage 2 reject  applied, but the sweep was rejected at the
#                                    top
#   Applied, undone below the top    applied, but an intermediate level threw
#                                    out every move in the sweep, or the
#                                    finished sweep was barred
#   Stage 1 reject                   screened out by its own block
#   No valid move                    barred, inadmissible, or nothing to draw
#
# SMC has no stage 2 (it reweights), so neither Reject nor Stage 1 accept +
# stage 2 reject occurs there.

TRACE_STATES = (
    ("Accept",                          "#22A822"),
    ("Reject",                          "#e34948"),
    ("Stage 1 accept + stage 2 reject", "#ff7f0e"),
    ("Applied, undone below the top",   "#54c8e2"),
    ("Stage 1 reject",                  "#ff9896"),
    ("No valid move",                   "#8c8578"),
)
(T_ACCEPT, T_REJECT, T_S2_REJECT, T_UNDONE, T_S1_REJECT,
 T_NO_MOVE) = range(len(TRACE_STATES))

# What each state means for a HINTS move, for the legend.
HINTS_MEANING = {
    T_ACCEPT:    "HINTS: move landed",
    T_S2_REJECT: "HINTS: sweep rejected at the top",
    T_UNDONE:    "HINTS: block applied it, a level above undid it",
    T_S1_REJECT: "HINTS: move screened out by its block",
}

SMC_TRACE_NOTE = ("SMC has no stage 2 -- it reweights -- so every proposal "
                  "that survives the proposal is kept, and Reject never occurs.")


def call_states(run):
    """For one .h5 run: the TRACE_STATES index of every sample() call, the
    call each logged move belongs to, and the TRACE_STATES index of every move."""
    call = run["call"][()]
    outcome = run["outcome"][()]
    call_outcome = run["call_outcome"][()]
    n_calls = len(call_outcome)
    # A stay is trivially "accepted" by the outer step, so the flag is only
    # read for calls that proposed something. SMC never rejects.
    accepted = (run["accepted"][()].astype(bool) if "accepted" in run
                else np.ones(n_calls, dtype=bool))

    # dsurr is nan exactly when a move was not screened.
    screened = np.bincount(call, weights=~np.isnan(run["dsurr"][()]), minlength=n_calls) > 0
    reached_screen = np.bincount(call, weights=(outcome == SCREENED_OUT) | (outcome == APPLIED),
                                 minlength=n_calls) > 0

    proposed = call_outcome == PROPOSED
    state = np.full(n_calls, T_NO_MOVE)
    state[proposed & accepted] = T_ACCEPT
    state[proposed & ~accepted & ~screened] = T_REJECT
    state[proposed & ~accepted & screened] = T_S2_REJECT
    state[call_outcome == SCREENED_OUT] = T_S1_REJECT               # DA
    # HINTS: a sweep with nothing left to propose is a STAY. If any of its
    # moves met a screen, the screens (or a level above them) emptied it.
    state[(call_outcome == STAY) & reached_screen] = T_S1_REJECT

    # A block-applied move's fate is its sweep's: it lands only if the sweep
    # does, and is lost at the top if the sweep is rejected there.
    move_state = np.full(len(call), T_NO_MOVE)
    move_state[outcome == SCREENED_OUT] = T_S1_REJECT
    applied = outcome == APPLIED
    sweep = state[call]
    move_state[applied] = T_UNDONE
    move_state[applied & (sweep == T_ACCEPT)] = T_ACCEPT
    move_state[applied & (sweep == T_S2_REJECT)] = T_S2_REJECT
    return state, call, move_state


def outcome_traces(experiment):
    """(states, moves, n_runs): per iteration (MCMC) or step (SMC), summed over
    chains, the (len(TRACE_STATES), n) counts of calls in each state and of
    moves in each state."""
    with h5py.File(experiment["h5_path"], "r") as f:
        runs = _run_names(f)
        per_run = []
        for name in runs:
            run = f[name]
            state, call, move_state = call_states(run)
            if "calls_at_step" in run:
                # SMC: one call per particle per step, in step order.
                step = np.searchsorted(run["calls_at_step"][()], np.arange(len(state)), side="right")
            else:
                step = np.arange(len(state))
            per_run.append((step, state, step[call], move_state))

    n = max(step.max() for step, *_ in per_run) + 1
    size = len(TRACE_STATES) * n
    states = np.zeros((len(TRACE_STATES), n))
    moves = np.zeros((len(TRACE_STATES), n))
    for step, state, move_step, move_state in per_run:
        states += np.bincount(state * n + step, minlength=size).reshape(-1, n)
        moves += np.bincount(move_state * n + move_step, minlength=size).reshape(-1, n)
    return states, moves, len(runs)


def _smooth(y, window):
    """Centred moving average along the last axis, averaging over only the
    points that exist at the edges."""
    if window <= 1:
        return y
    kernel = np.ones(window)
    num = np.apply_along_axis(np.convolve, -1, y, kernel, "same")
    return num / np.convolve(np.ones(y.shape[-1]), kernel, "same")


def _stack_figure(panels, title, legend_meaning, footnote, ncols):
    """One figure of 100% stacks. `panels` is [(experiment, counts, subtitle,
    ylabel, xlabel)], counts a (len(TRACE_STATES), n) array per iteration.
    Only the states that occur go in the legend, each followed by its entry
    in `legend_meaning` where it has one."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = _panel_grid(len(panels), ncols, 5.4, 3.8, sharey=True)
    for ax, (experiment, counts, subtitle, ylabel, xlabel) in zip(axes, panels):
        n = counts.shape[1]
        share = 100.0 * counts / np.maximum(counts.sum(axis=0), 1e-12)
        ax.stackplot(np.arange(n), share, colors=[c for _, c in TRACE_STATES], linewidth=0, alpha=0.75)
        ax.set_xlim(0, max(n - 1, 1))
        ax.set_ylim(0, 100)
        ax.set_title(f"{panel_title(experiment)}\n{subtitle}", fontsize=12, fontweight="bold")
        if ax.get_subplotspec().rowspan.start == nrows - 1:
            ax.set_xlabel(xlabel)
        if ax.get_subplotspec().colspan.start == 0:
            ax.set_ylabel(ylabel)

    handles = []
    for k, (name, colour) in enumerate(TRACE_STATES):
        if not any(p[1][k].any() for p in panels):
            continue
        if k in legend_meaning:
            name = f"{name} ({legend_meaning[k]})"
        handles.append(Patch(facecolor=colour, label=name))
    n_rows = 1 if len(handles) <= 4 else 2
    fig.legend(handles=handles, loc="upper center", frameon=False,
               ncol=max(1, int(np.ceil(len(handles) / n_rows))), bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.97)
    if footnote:
        fig.text(0.5, -0.01, footnote, ha="center", va="top", fontsize=10, fontstyle="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97 - 0.02 * (n_rows - 1)))
    plt.show()


def plot_outcome_traces(results, samplername, smooth=0.005, ncols=3, footnote=None):
    """Two figures of 100% stacks per iteration: the share of calls in each
    TRACE_STATES state for every experiment, then, for (Flat) HINTS, the
    share of every move drawn in the sweeps. `smooth` is the moving-average
    window as a fraction of each run's length (0 for none)."""
    traces = [outcome_traces(e) for e in results]
    step_name = "Step" if samplername == "SMC" else "Iteration"
    unit = "particles" if samplername == "SMC" else "chains"

    calls, moves = [], []
    for experiment, (states, move_counts, n_runs) in zip(results, traces):
        n = states.shape[1]
        window = max(1, int(round(smooth * n)))
        subtitle = f"{n_runs} runs, window {window}"
        calls.append((experiment, _smooth(states, window), subtitle, f"Share of {unit} (%)", step_name))
        if "HINTS" in experiment["proposal_mechanism"]:
            per_sweep = move_counts.sum() / states.sum()
            moves.append((experiment, _smooth(move_counts, window),
                          f"{subtitle}, {per_sweep:.1f} moves per sweep",
                          "Share of moves drawn (%)", step_name))

    _stack_figure(calls, f"Outcome of each proposal per {step_name.lower()} -- "
                         f"{samplername} experiments on {results[0]['dataset']}",
                  {}, footnote, ncols)
    if not moves:
        return
    notes = [footnote] if footnote else []
    notes.append(
        "Every move drawn in the sweeps, not every sweep. A move applied by its block lands "
        "only if its sweep is accepted at the top. Hierarchical HINTS does not log\nwhich "
        "moves an intermediate level undid in a sweep that still went on, so its Accept share "
        "is an upper bound and its \"undone below the top\" share a lower one; for Flat "
        "HINTS both are exact.")
    _stack_figure(moves, f"Outcome of each move in a HINTS sweep per {step_name.lower()} -- "
                         f"{samplername} experiments on {results[0]['dataset']}",
                  HINTS_MEANING, "\n".join(notes), ncols)
