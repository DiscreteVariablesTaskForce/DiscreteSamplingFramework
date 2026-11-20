"""
Compare the samplers of one or more runs at equal row-read budgets: the
analysis behind Chapters 3 (MCMC) and 4 (SMC).

    python examples/incremental_decision_tree/analyse_results.py \
        --run-ids ch34 hints_2x4 --reference reference
    python examples/incremental_decision_tree/analyse_results.py --run-ids ch3_budget

Run it from the repo root once evaluate_results.py has written the
*_metrics.npz files. Under --out (default Results/analysis) it writes

    tables/<dataset>_mcmc.{csv,tex}         every MCMC sampler over its own run
    tables/<dataset>_mcmc_budget.{csv,tex}  every MCMC sampler at one common budget
    tables/<dataset>_smc.{csv,tex}
    tables/<dataset>_moves.{csv,tex}        what became of the moves proposed
    tables/<dataset>_reference.csv          the reference chains, one row each
    figures/<dataset>_*.pdf
    cache/<run-id>/<stem>.pkl               what was read off each .h5, so a
                                            rerun takes seconds

Cost is counted in row reads (problem.row_reads), in place of Chapter 2's full
evaluations: scoring a tree is cheap, and reading rows is what a sampler pays
for. The measures follow Chapter 2's definitions with that substitution:

  log posterior   The runs store trees, not their scores, so each stored tree's
                  log posterior is recomputed here from its splits and leaf
                  counts: the same sum IncrementalTreeTarget.eval makes.
  level           The lowest final-quarter mean log posterior among the
                  reference chains: a level every long MH chain had passed by
                  the end of its run.
  burn-in         The row reads a chain uses before its log posterior first
                  reaches the level. A chain that never does is censored, and
                  the table says how many reached it.
  second half     The records after half of a run's row reads, over which the
                  mixing, R-hat and prediction measures are taken.
  IACT            1 + 2 sum_{k<=M} rho(k), with Sokal's adaptive window, the
                  smallest M >= 5 tau, computed on the stored records and
                  converted to iterations by the iterations per record. inf
                  when the window never closes, i.e. the run is too short to
                  measure it.
  reads per independent state
                  The second half's row reads divided by its effective sample
                  size, iterations / IACT.
  R-hat           sqrt((W + B) / W) over the chains' second halves, W the mean
                  within-chain variance and B the variance of the chain means.

The reference chains share their seeds with the MH chains of the main run, so
they are those chains continued, not independent of them.
"""
import argparse
import hashlib
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discretesampling.domain.incremental_decision_tree import diagnostics as dg  # noqa: E402
from discretesampling.domain.incremental_decision_tree.target import log_catalan  # noqa: E402
from plot_diagnostics import ACCEPTED_FATES, FATES, move_fates  # noqa: E402
from results_io import (LEGACY_INIT_TREE_SIZE, PROPOSALS, available_cpus,  # noqa: E402
                        experiment_filename, init_of, iter_experiment_hdf5,
                        metrics_filename, read_run_config)
from sampler_diagnostics import DEFAULT_CFG, load_data  # noqa: E402

# Bump when a summary changes shape, so stale caches are rebuilt.
CACHE_VERSION = 2

# The names proposals are shown under, and their colours. Colour follows the
# proposal (and HINTS' shape), never its position in a plot; the long MH
# reference is drawn in muted ink as the baseline every panel shares.
PROPOSAL_NAMES = {"MH": "MH", "DA": "DA", "FlatHINTS": "Flat HINTS", "HINTS": "HINTS"}
COLOURS = {"HINTS": "#2a78d6", "Flat HINTS": "#eb6834", "DA": "#1baf7a",
           "HINTS 2×4": "#eda100", "MH": "#52514e", "MH (reference)": "#52514e"}
GRID_INK = "#d9d8d4"
# Figure titles; --no-titles drops them for figures captioned elsewhere.
SHOW_TITLES = True
MUTED_INK = "#898781"

# The move-log fates (plot_diagnostics.FATES) grouped the way the tables
# report them.
FATE_GROUPS = {
    "accepted": ACCEPTED_FATES,
    "screened out": (3, 4),
    "rejected": (2, 5),
    "barred": (6,),
}
STAY_FATE = 7
assert len(FATES) == 8


# --------------------------------------------------------------------------- #
# what was run
# --------------------------------------------------------------------------- #

@dataclass
class Experiment:
    run_id: str
    dataset: str
    sampler: str
    proposal: str
    ss_prop: float
    shape: str | None           # "<levels>x<branching>", HINTS only
    reference: bool
    path: str
    metrics_path: str
    store_every: int
    lam: float
    separate: bool = False      # the reference comes from a run of its own
    runs: list = field(default_factory=list, repr=False)
    metrics: dict = field(default_factory=dict, repr=False)

    @property
    def label(self):
        if self.proposal == "MH":
            return "MH (reference)" if self.reference and self.separate else "MH"
        name = PROPOSAL_NAMES[self.proposal]
        if self.proposal == "HINTS" and self.shape and self.shape != binary_shape(self.ss_prop):
            name += " " + self.shape.replace("x", "×")
        return name

    @property
    def ss_label(self):
        return "1" if self.proposal == "MH" else f"1/{round(1 / self.ss_prop)}"

    @property
    def colour(self):
        return COLOURS[self.label]

    def sort_key(self):
        return (self.reference, PROPOSALS.index(self.proposal),
                self.label != PROPOSAL_NAMES[self.proposal], -self.ss_prop)


def binary_shape(ss_prop):
    """The default HINTS hierarchy at ss_prop: binary levels down to blocks of it."""
    return f"{int(round(np.log2(1 / ss_prop)))}x2"


def _as_list(value):
    return [value] if isinstance(value, str) else list(value)


def discover(results_root, run_ids, reference=None):
    """Every (sampler, proposal) file the runs' config.json files list, the
    reference run's included when one is named."""
    out = []
    for run_id in [*run_ids, *([reference] if reference else [])]:
        results_dir = os.path.join(results_root, run_id)
        config = read_run_config(results_dir)
        entries = config.get("experiments") or ([config] if "dataset" in config else [])
        if not entries:
            raise SystemExit(f"{results_dir} has no config.json listing its experiments")
        for entry in entries:
            cfg = {**DEFAULT_CFG, "init_tree_size": LEGACY_INIT_TREE_SIZE, **entry}
            for sampler in _as_list(cfg["samplers"]):
                for proposal in _as_list(cfg["proposals"]):
                    args = (cfg["dataset"], sampler, proposal, cfg["ss_prop"], init_of(cfg))
                    path = os.path.join(results_dir, experiment_filename(*args))
                    if not os.path.exists(path):
                        print(f"[warning] {path} is listed in config.json but missing; skipped")
                        continue
                    shape = (f"{entry['levels']}x{entry['branching']}"
                             if proposal == "HINTS" and "levels" in entry else None)
                    out.append(Experiment(
                        run_id=run_id, dataset=cfg["dataset"], sampler=sampler,
                        proposal=proposal,
                        ss_prop=1.0 if proposal == "MH" else float(cfg["ss_prop"]),
                        shape=shape, reference=run_id == reference,
                        separate=run_id == reference, path=path,
                        metrics_path=os.path.join(results_dir, metrics_filename(*args)),
                        store_every=int(cfg["store_every"]), lam=float(cfg["lam"])))
    return sorted(out, key=Experiment.sort_key)


# --------------------------------------------------------------------------- #
# the stored trees' log posteriors
# --------------------------------------------------------------------------- #

_PRIORS = {}


def split_priors(dataset):
    """(lp_vals, lp_feats) of the problem the dataset's runs were sampled on:
    IncrementalTreeProblem's uniform feature and threshold priors, over the
    training split's ranges."""
    if dataset not in _PRIORS:
        X_train = load_data(dataset)[0]
        lp_vals = -np.log(X_train.max(axis=0).astype(np.float64)
                          - X_train.min(axis=0).astype(np.float64))
        _PRIORS[dataset] = (lp_vals, -np.log(X_train.shape[1]))
    return _PRIORS[dataset]


def segment_sums(values, lengths):
    """Sums of consecutive segments of `values`, of the given lengths (0 allowed)."""
    total = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    end = np.cumsum(lengths)
    return total[end] - total[end - lengths]


def stored_states(run, lam, lp_vals, lp_feats):
    """
    One entry per tree a run stored: its log posterior, node count, root
    feature (-1 for a stump) and threshold, and a key that two stored trees
    share exactly when they are the same tree.

    The log posterior is IncrementalTreeTarget.eval's sum: the Poisson prior on
    the node count, less the log Catalan number of the leaf count, the uniform
    feature and threshold priors of every split, and the Dirichlet-multinomial
    likelihood of every leaf's counts. A stored tree is a state a sampler held,
    so it is always valid and the out-of-support value never arises.
    """
    nodes = np.asarray(run["state_tree_lengths"], dtype=np.int64)
    leaves = np.asarray(run["state_leaf_lengths"], dtype=np.int64)
    empty_i, empty_f = np.zeros(0, np.int64), np.zeros(0, np.float64)
    feat = np.asarray(run.get("state_tree_feat", empty_i), dtype=np.int64)
    thr = np.asarray(run.get("state_tree_thr", empty_f), dtype=np.float64)
    depth = np.asarray(run.get("state_tree_depth", empty_i), dtype=np.int64)
    counts = np.asarray(run["state_leaf_counts"], dtype=np.float64)
    alpha = float(run.get("state_alpha", 1.0))
    K = counts.shape[1]

    a0 = alpha * K
    leaf_dm = (gammaln(a0) - gammaln(a0 + counts.sum(axis=1))
               + gammaln(counts + alpha).sum(axis=1) - K * gammaln(alpha))
    log_lik = segment_sums(leaf_dm, leaves)
    catalan = np.array([log_catalan(n) for n in range(int(leaves.max()) + 1)])
    log_prior = (nodes * np.log(lam) - lam - gammaln(nodes + 1) - catalan[leaves]
                 + nodes * lp_feats + segment_sums(lp_vals[feat], nodes))

    owner = np.repeat(np.arange(len(nodes)), nodes)
    root = np.flatnonzero(depth == 0)
    root_feat = np.full(len(nodes), -1, dtype=np.int64)
    root_thr = np.full(len(nodes), np.nan)
    root_feat[owner[root]] = feat[root]
    root_thr[owner[root]] = thr[root]

    # A tree is its set of (threshold, feature) splits: thresholds are drawn
    # from a continuous range, so two different trees never share them all.
    starts = np.concatenate([[0], np.cumsum(nodes)])
    keys = np.empty(len(nodes), dtype=np.int64)
    for s in range(len(nodes)):
        a, b = starts[s], starts[s + 1]
        order = np.argsort(thr[a:b], kind="stable")
        digest = hashlib.blake2b(thr[a:b][order].tobytes() + feat[a:b][order].tobytes(),
                                 digest_size=8).digest()
        keys[s] = int.from_bytes(digest, "little", signed=True)
    return log_prior + log_lik, nodes, root_feat, root_thr, keys


# --------------------------------------------------------------------------- #
# reading a file down to what the analysis needs
# --------------------------------------------------------------------------- #

def _counters(run):
    return {k: float(v) for k, v in run.items()
            if (k.startswith("prop_") or k.startswith("target_")) and np.ndim(v) == 0}


def summarise_mcmc_run(run, lam, lp_vals, lp_feats):
    """
    One chain, read down to its records. A run given a budget of row reads
    keeps its columns at its records only, alongside an `iteration` column and
    run-level counts; a run set by iterations keeps them every iteration, and
    its counts of full evaluations are worked out from those.
    """
    lp, nodes, root_feat, root_thr, keys = stored_states(run, lam, lp_vals, lp_feats)
    slot = np.asarray(run["state_index"])[:, 0]
    it = np.asarray(run["state_iters"], dtype=np.int64)
    reads = np.asarray(run["row_reads"], dtype=np.int64)
    if "iteration" in run:
        if not np.array_equal(np.asarray(run["iteration"]), it):
            raise ValueError("a budget run's columns do not line up with its records")
        iters, rec_reads = int(run["iterations"]), reads
        full, accepted_full = int(run["full_evals"]), int(run["accepted_full"])
    else:
        iters, rec_reads = len(reads), reads[it]
        full = accepted_full = None
        if "call_outcome" in run:
            proposed = np.asarray(run["call_outcome"]) == dg.PROPOSED
            full = int(proposed.sum())
            accepted_full = int((proposed & np.asarray(run["accepted"]).astype(bool)).sum())
    return dict(
        iters=iters, total_reads=int(reads[-1]),
        seconds=float(run["cumulative_time"][-1]),
        rec_it=it, rec_reads=rec_reads, rec_lp=lp[slot], rec_nodes=nodes[slot],
        rec_root_feat=root_feat[slot], rec_root_thr=root_thr[slot], rec_key=keys[slot],
        acceptance_rate=float(run["acceptance_rate"]),
        full_evals=full, accepted_full=accepted_full,
        budget=float(run["budget"]) if "budget" in run else None,
        fates=move_fates(run), counters=_counters(run))


def summarise_smc_run(run, lam, lp_vals, lp_feats):
    lp, nodes, _, _, keys = stored_states(run, lam, lp_vals, lp_feats)
    slots = np.asarray(run["state_index"])
    logw = np.asarray(run["state_log_weights"], dtype=np.float64)
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    reads = np.asarray(run["row_reads"], dtype=np.int64)
    it = np.asarray(run["state_iters"], dtype=np.int64)
    return dict(
        steps=len(reads), particles=slots.shape[1], total_reads=int(reads[-1]),
        seconds=float(run["cumulative_time"][-1]),
        # ess[t] is the ESS the weights had at the start of step t, before
        # any resampling: ess[0] is the prior start's, ess[t > 0] what step
        # t - 1's moves left.
        ess=np.asarray(run["ess"], dtype=np.float64),
        resampled=np.asarray(run["resampled"], dtype=bool), reads=reads,
        rec_it=it, rec_reads=reads[it],
        rec_wlp=(w * lp[slots]).sum(axis=1), rec_maxlp=lp[slots].max(axis=1),
        rec_wnodes=(w * nodes[slots]).sum(axis=1),
        rec_distinct=np.array([len(np.unique(keys[s])) for s in slots]),
        rec_ess=1.0 / (w ** 2).sum(axis=1),
        fates=move_fates(run), counters=_counters(run))


def _cache_path(cache_root, experiment):
    stem = os.path.splitext(os.path.basename(experiment.path))[0]
    return os.path.join(cache_root, experiment.run_id, stem + ".pkl")


def _stamp(experiment):
    source = os.stat(experiment.path)
    return dict(version=CACHE_VERSION, mtime_ns=source.st_mtime_ns, size=source.st_size)


def summarise_file(job):
    """Read one .h5 into its per-run summaries, through the cache."""
    experiment, cache_root, refresh = job
    cache = _cache_path(cache_root, experiment)
    stamp = _stamp(experiment)
    if not refresh and os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                saved = pickle.load(f)
            if saved["stamp"] == stamp:
                return saved["runs"]
        except (OSError, EOFError, pickle.UnpicklingError, KeyError):
            pass
    lp_vals, lp_feats = split_priors(experiment.dataset)
    summarise = summarise_mcmc_run if experiment.sampler == "mcmc" else summarise_smc_run
    runs = [summarise(run, experiment.lam, lp_vals, lp_feats)
            for run in iter_experiment_hdf5(experiment.path)]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = f"{cache}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dict(stamp=stamp, runs=runs), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache)
    return runs


def load_metrics(experiment, keys=("test_log_loss", "test_accuracy")):
    """{metric: (n_runs, n_records)} off the experiment's _metrics.npz, checked to
    line up with the stored records; {} if it has not been evaluated."""
    if not os.path.exists(experiment.metrics_path):
        print(f"[warning] {experiment.metrics_path} missing: run evaluate_results.py; "
              f"prediction measures left blank for {experiment.label} {experiment.ss_label}")
        return {}
    with np.load(experiment.metrics_path) as saved:
        if int(saved["stride"]) != 1:
            raise SystemExit(f"{experiment.metrics_path} was evaluated at --stride "
                             f"{int(saved['stride'])}; this analysis needs every record")
        iterations = saved["iterations"]
        # A run with a budget records at the same row reads in every chain,
        # but at different iterations, so only the record count can be checked.
        lines_up = (len(iterations) == len(experiment.runs[0]["rec_it"])
                    if experiment.runs[0].get("budget") else
                    np.array_equal(iterations, experiment.runs[0]["rec_it"]))
        if not lines_up:
            raise SystemExit(f"{experiment.metrics_path} does not line up with the "
                             f"records of {experiment.path}")
        return {k: saved[k] for k in keys if k in saved.files}


# --------------------------------------------------------------------------- #
# measures
# --------------------------------------------------------------------------- #

def iact(x):
    """
    Integrated autocorrelation time of one series, in its own steps, with
    Sokal's adaptive window: 1 + 2 sum_{k=1}^{M} rho(k) at the smallest M with
    M >= 5 tau. nan for a constant series, inf when the window never closes.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4 or np.ptp(x) == 0:
        return np.nan
    x = x - x.mean()
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    acf /= acf[0]
    tau = 1.0 + 2.0 * np.cumsum(acf[1:])
    closed = np.arange(1, n) >= 5.0 * tau
    return float(tau[np.argmax(closed)]) if closed.any() else np.inf


def rhat(series):
    """sqrt((W + B) / W) over the given chains; nan if every chain is constant."""
    series = [np.asarray(s, dtype=np.float64) for s in series if len(s) > 1]
    if len(series) < 2:
        return np.nan
    W = np.mean([s.var(ddof=1) for s in series])
    B = np.var([s.mean() for s in series], ddof=1)
    return float(np.sqrt((W + B) / W)) if W > 0 else np.nan


def second_half(run):
    """Mask of a run's records after half of its row reads."""
    return run["rec_reads"] >= run["total_reads"] / 2


def value_at(reads, values, budget):
    """A run's value at a row-read budget: its last record at or below it; nan
    if the run stopped short of the budget or had no record by then."""
    if budget > reads[-1]:
        return np.nan
    j = np.searchsorted(reads, budget, side="right") - 1
    return values[j] if j >= 0 else np.nan


def reads_to_reach(run, level):
    """Row reads before a chain's log posterior first reaches `level`; inf if never."""
    hit = np.flatnonzero(run["rec_lp"] >= level)
    return float(run["rec_reads"][hit[0]]) if len(hit) else np.inf


def nanmedian(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(np.median(values)) if len(values) else np.nan


def reference_level(reference):
    """The lowest final-quarter mean log posterior over the reference chains."""
    means = [r["rec_lp"][r["rec_reads"] >= 0.75 * r["total_reads"]].mean()
             for r in reference.runs]
    return float(min(means)), means


def mixing(experiment, run_index, values):
    """
    (IACT in iterations, ESS, reads per independent state) of one chain's
    second half of `values`. The IACT is taken over the records, whether they
    are evenly spaced in iterations or in row reads, so the effective sample
    size is the records over it; it is converted to iterations by the
    iterations per record over the half.
    """
    run = experiment.runs[run_index]
    half = np.flatnonzero(second_half(run))
    tau_records = iact(values[half])
    ess = len(half) / tau_records if tau_records > 0 else np.nan
    its = run["rec_it"][half]
    per_record = (its[-1] - its[0]) / max(len(half) - 1, 1)
    reads = run["total_reads"] - run["rec_reads"][half[0]]
    return tau_records * per_record, ess, reads / ess if ess > 0 else np.inf


def mcmc_row(experiment, level, reference):
    runs = experiment.runs
    out = dict(sampler=experiment.label, ss=experiment.ss_label)
    out["iters"] = runs[0]["iters"]
    out["reads_per_iter"] = nanmedian([r["total_reads"] / r["iters"] for r in runs])
    out["seconds_per_iter"] = nanmedian([r["seconds"] / r["iters"] for r in runs])
    out["total_reads"] = nanmedian([r["total_reads"] for r in runs])
    burn = [reads_to_reach(r, level) for r in runs]
    out["burnin_reads"] = float(np.median(burn))
    out["burnin_reached"] = int(np.isfinite(burn).sum())
    out["final_log_post"] = nanmedian([r["rec_lp"][-1] for r in runs])
    out["acceptance_rate"] = nanmedian([r["acceptance_rate"] for r in runs])

    series = {"log_post": [r["rec_lp"] for r in runs],
              "nodes": [r["rec_nodes"].astype(np.float64) for r in runs]}
    if "test_log_loss" in experiment.metrics:
        series["log_loss"] = list(experiment.metrics["test_log_loss"])
    for name, values in series.items():
        stats = np.array([mixing(experiment, i, v) for i, v in enumerate(values)])
        out[f"iact_{name}"] = nanmedian(stats[:, 0])
        out[f"reads_per_indep_{name}"] = nanmedian(stats[:, 2])
        out[f"rhat_{name}"] = rhat([v[second_half(r)] for r, v in zip(runs, values)])

    out["nodes"] = float(np.concatenate([r["rec_nodes"][second_half(r)] for r in runs]).mean())
    for key, name in (("test_log_loss", "log_loss"), ("test_accuracy", "accuracy")):
        if key in experiment.metrics:
            out[name] = float(np.concatenate(
                [m[second_half(r)] for r, m in zip(runs, experiment.metrics[key])]).mean())
            if reference is not None and key in reference.metrics:
                ref = float(np.concatenate(
                    [m[second_half(r)] for r, m in zip(reference.runs, reference.metrics[key])]).mean())
                out[f"{name}_vs_ref"] = out[name] - ref
    return out


def budget_row(experiment, budget, level):
    """One MCMC sampler at a common row-read budget: where its chains are at
    it, and their prediction and exploration over its second half."""
    runs = experiment.runs
    out = dict(sampler=experiment.label, ss=experiment.ss_label, budget=budget)
    out["log_post"] = nanmedian([value_at(r["rec_reads"], r["rec_lp"], budget) for r in runs])
    out["reached"] = int(sum(reads_to_reach(r, level) <= budget for r in runs))
    out["chains_at_budget"] = int(sum(r["total_reads"] >= budget for r in runs))
    window = [(r["rec_reads"] > budget / 2) & (r["rec_reads"] <= budget) for r in runs]
    out["nodes"] = nanmedian([r["rec_nodes"][w].mean() if w.any() else np.nan
                              for r, w in zip(runs, window)])
    for key, name in (("test_log_loss", "log_loss"), ("test_accuracy", "accuracy")):
        if key in experiment.metrics:
            out[name] = nanmedian([m[w].mean() if w.any() else np.nan
                                   for m, w in zip(experiment.metrics[key], window)])
    upto = [r["rec_reads"] <= budget for r in runs]
    out["root_features"] = nanmedian([len(np.unique(r["rec_root_feat"][u])) for r, u in zip(runs, upto)])
    out["root_splits"] = nanmedian([len(np.unique(r["rec_root_thr"][u])) for r, u in zip(runs, upto)])
    out["distinct_trees"] = nanmedian([len(np.unique(r["rec_key"][u])) for r, u in zip(runs, upto)])
    return out


def common_budgets(experiments):
    """The powers of ten of row reads that at least half the chains of at
    least two samplers reach: the budgets the samplers are compared at."""
    totals = [np.median([r["total_reads"] for r in e.runs]) for e in experiments]
    lo = int(np.ceil(np.log10(min(totals))))
    hi = int(np.floor(np.log10(max(totals))))
    return [10.0 ** k for k in range(lo - 1, hi + 1)
            if sum(t >= 10.0 ** k for t in totals) >= 2]


def smc_row(experiment, reference):
    runs = experiment.runs
    out = dict(sampler=experiment.label, ss=experiment.ss_label)
    out["particles"] = runs[0]["particles"]
    out["steps"] = runs[0]["steps"]
    out["total_reads"] = nanmedian([r["total_reads"] for r in runs])
    out["reads_per_step"] = nanmedian([r["total_reads"] / r["steps"] for r in runs])
    out["seconds_per_step"] = nanmedian([r["seconds"] / r["steps"] for r in runs])
    out["start_ess"] = nanmedian([r["ess"][0] for r in runs])
    out["median_ess_frac"] = nanmedian([np.median(r["ess"][1:]) / r["particles"] for r in runs])
    out["resample_rate"] = nanmedian([r["resampled"].mean() for r in runs])
    out["final_log_post"] = nanmedian([r["rec_wlp"][-1] for r in runs])
    out["final_distinct"] = nanmedian([r["rec_distinct"][-1] for r in runs])
    out["final_nodes"] = nanmedian([r["rec_wnodes"][-1] for r in runs])
    for key, name in (("test_log_loss", "log_loss"), ("test_accuracy", "accuracy")):
        if key in experiment.metrics:
            out[name] = nanmedian(experiment.metrics[key][:, -1])
            if reference is not None and key in reference.metrics:
                ref = float(np.concatenate(
                    [m[second_half(r)] for r, m in zip(reference.runs, reference.metrics[key])]).mean())
                out[f"{name}_vs_ref"] = out[name] - ref
    return out


def moves_row(experiment):
    tables = [r["fates"] for r in experiment.runs if r["fates"] is not None]
    if not tables:
        return None
    table = np.sum(tables, axis=0)                      # (moves, fates)
    calls = sum(r.get("iters", r.get("steps", 0)) * r.get("particles", 1) for r in experiment.runs)
    stay = dg.MOVES.index("stay")
    real = np.delete(np.arange(len(dg.MOVES)), stay)
    moves = table[real].sum()
    out = dict(sampler=experiment.label, kind=experiment.sampler.upper(), ss=experiment.ss_label)
    out["moves_per_call"] = moves / calls
    for name in ("grow", "prune", "change"):
        i = dg.MOVES.index(name)
        out[f"share_{name}"] = table[i].sum() / moves
        out[f"accept_{name}"] = table[i, list(ACCEPTED_FATES)].sum() / max(table[i].sum(), 1)
    for name, fates in FATE_GROUPS.items():
        out[name] = table[real][:, list(fates)].sum() / moves
    out["stay_calls"] = table[stay, STAY_FATE] / calls
    return out


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def fmt_sci(x):
    if not np.isfinite(x):
        return "--" if np.isnan(x) else r"$\infty$"
    mantissa, exponent = f"{x:.2e}".split("e")
    exponent = int(exponent)
    if -2 <= exponent <= 3:
        return f"{x:,.3g}".replace(",", "{,}")
    return rf"${float(mantissa):.2g}\times10^{{{exponent}}}$"


def fmt_num(digits):
    def f(x):
        if isinstance(x, str):
            return x
        if not np.isfinite(x):
            return "--" if np.isnan(x) else r"$\infty$"
        return f"{x:,.{digits}f}".replace(",", "{,}")
    return f


def fmt_pct(x):
    return "--" if not np.isfinite(x) else f"{100 * x:.1f}"


def fmt_signed(digits):
    def f(x):
        return "--" if not np.isfinite(x) else f"{x:+.{digits}f}"
    return f


def write_table(path, rows, columns, caption):
    """rows as CSV (raw values, every key) and as a booktabs LaTeX table of
    `columns`, [(key, header, formatter)]."""
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path + ".csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(_csv(row.get(k, "")) for k in keys) + "\n")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" * 2 + "r" * (len(columns) - 2) + "}", r"\toprule",
             " & ".join(h for _, h, _ in columns) + r" \\", r"\midrule"]
    for row in rows:
        cells = [fmt(row[k]) if k in row else "--" for k, _, fmt in columns]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}",
              rf"\label{{tab:{os.path.basename(path)}}}", r"\end{table}"]
    with open(path + ".tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def _csv(value):
    if isinstance(value, str):
        return value
    return repr(float(value)) if np.ndim(value) == 0 else ""


def text(x):
    return x


MCMC_COLUMNS = [
    ("sampler", "Sampler", text), ("ss", "$\\omega$", text),
    ("reads_per_iter", "Reads/iter.", fmt_sci),
    ("total_reads", "Reads", fmt_sci),
    ("burnin_reads", "Burn-in", fmt_sci),
    ("burnin_reached", "Reached", lambda x: f"{x}/16"),
    ("final_log_post", r"$\log\pi$ final", fmt_sci),
    ("reads_per_indep_log_post", r"Reads/indep.\ ($\log\pi$)", fmt_sci),
    ("reads_per_indep_log_loss", "Reads/indep.\\ (loss)", fmt_sci),
    ("rhat_log_post", r"$\hat R$ ($\log\pi$)", fmt_num(2)),
    ("nodes", "Nodes", fmt_num(1)),
    ("log_loss", "Log loss", fmt_num(4)),
    ("accuracy", "Acc.", fmt_num(3)),
]
BUDGET_COLUMNS = [
    ("sampler", "Sampler", text), ("ss", "$\\omega$", text),
    ("budget", "Budget", fmt_sci),
    ("chains_at_budget", "Chains", fmt_num(0)),
    ("log_post", r"$\log\pi$ at budget", fmt_sci),
    ("reached", "Reached", fmt_num(0)),
    ("nodes", "Nodes", fmt_num(1)),
    ("log_loss", "Log loss", fmt_num(4)),
    ("accuracy", "Acc.", fmt_num(3)),
    ("root_features", "Root feat.", fmt_num(0)),
    ("root_splits", "Root splits", fmt_num(0)),
]
SMC_COLUMNS = [
    ("sampler", "Sampler", text), ("ss", "$\\omega$", text),
    ("reads_per_step", "Reads/step", fmt_sci),
    ("total_reads", "Reads", fmt_sci),
    ("start_ess", "ESS$_0$", fmt_num(1)),
    ("median_ess_frac", "ESS/$N$", fmt_num(2)),
    ("resample_rate", "Resampled", fmt_num(2)),
    ("final_log_post", r"$\log\pi$ final", fmt_sci),
    ("final_distinct", "Trees", fmt_num(0)),
    ("final_nodes", "Nodes", fmt_num(1)),
    ("log_loss", "Log loss", fmt_num(4)),
    ("accuracy", "Acc.", fmt_num(3)),
]
MOVES_COLUMNS = [
    ("sampler", "Sampler", text), ("kind", "", text), ("ss", "$\\omega$", text),
    ("moves_per_call", "Moves/call", fmt_num(1)),
    ("accept_grow", "Grow acc.\\ \\%", fmt_pct),
    ("accept_prune", "Prune acc.\\ \\%", fmt_pct),
    ("accept_change", "Change acc.\\ \\%", fmt_pct),
    ("accepted", "Accepted \\%", fmt_pct),
    ("screened out", "Screened out \\%", fmt_pct),
    ("rejected", "Rejected \\%", fmt_pct),
    ("barred", "Barred \\%", fmt_pct),
]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED_INK, "axes.grid": True, "grid.color": GRID_INK,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def curve_on_grid(runs, x_key, y_key, grid):
    """(median, lower quartile, upper quartile) of the runs' y at each grid x,
    each run holding its last record's value; nan where under half the runs
    reach that x."""
    at = np.array([[value_at(r[x_key], r[y_key], g) for g in grid] for r in runs])
    present = (~np.isnan(at)).sum(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-nan columns
        q = np.nanpercentile(at, [50, 25, 75], axis=0)
    q[:, present < len(runs) / 2] = np.nan
    return q


def read_grid(experiments, key="rec_reads", decades=4):
    """A log grid of row reads over the last `decades` decades before the
    longest run's end: before that every run is still at its prior start."""
    hi = max(r[key][-1] for e in experiments for r in e.runs)
    return np.geomspace(hi / 10.0 ** decades, hi, 300)


def panels_by_ss(experiments):
    """{ss_prop: experiments} over the subsampling proposals' fractions."""
    out = {}
    for e in experiments:
        if e.proposal != "MH":
            out.setdefault(e.ss_prop, []).append(e)
    return dict(sorted(out.items(), reverse=True))


def _panel_axes(plt, n, width=3.4, height=2.5):
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows),
                             squeeze=False, sharey=True, sharex=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n]), cols


def _finish(plt, fig, axes, cols, path, title, handles=None):
    """One legend for the whole figure, below the panels, then save."""
    if handles is None:
        handles = {}
        for ax in axes:
            for h, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, h)
    if SHOW_TITLES:
        fig.suptitle(title, size=9)
    fig.tight_layout(rect=(0, 0.08 if len(axes) <= cols else 0.05, 1, 1))
    fig.legend(handles.values(), handles.keys(), loc="lower center",
               ncol=min(len(handles), 5), frameon=False)
    fig.savefig(path)
    plt.close(fig)


def _ss_title(ax, ss):
    ax.set_title(f"$\\omega = 1/{round(1 / ss)}$")


def plot_progress(plt, path, dataset, experiments, reference, y_key, ylabel, level=None,
                  ref_values=None, metrics_key=None, ref_value=None):
    """Median and interquartile range over the chains of a per-record value
    against row reads, one panel per subsample fraction, with the reference MH
    chains in every panel."""
    panels = panels_by_ss(experiments)
    if not panels:
        return
    grid = read_grid([e for es in panels.values() for e in es] + [reference])
    fig, axes, cols = _panel_axes(plt, len(panels))
    for ax, (ss, group) in zip(axes, panels.items()):
        for e in [reference] + group:
            runs, key = e.runs, y_key
            if metrics_key is not None:
                if metrics_key not in e.metrics:
                    continue
                runs = [dict(r, value=m) for r, m in zip(e.runs, e.metrics[metrics_key])]
                key = "value"
            med, lo, hi = curve_on_grid(runs, "rec_reads", key, grid)
            ax.plot(grid, med, "--" if e.reference else "-", color=e.colour, label=e.label)
            ax.fill_between(grid, lo, hi, color=e.colour, alpha=0.15, linewidth=0)
        for value in [level, ref_value]:
            if value is not None:
                ax.axhline(value, color=MUTED_INK, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        _ss_title(ax, ss)
    if level is not None and ref_values:
        # Down to three times the reference chains' spread below the level:
        # the approach to it, not the climb from the prior.
        spread = max(ref_values) - level
        axes[0].set_ylim(bottom=level - 3 * spread)
    for i, ax in enumerate(axes):
        if i % cols == 0:
            ax.set_ylabel(ylabel)
        if i >= len(axes) - cols:
            ax.set_xlabel("Row reads")
    note = "; dotted: the level" if level is not None else "; dotted: the reference's second-half mean"
    _finish(plt, fig, axes, cols, path,
            f"{dataset}: median over 16 chains, interquartile range shaded{note}")


SS_MARKERS = {4: "o", 8: "s", 16: "^", 32: "D"}


def plot_mixing(plt, path, dataset, rows_by_experiment):
    """Mixing against cost: IACT of the log posterior per iteration against row
    reads per iteration, colour the proposal and marker the subsample, with
    grey lines of equal reads per independent state."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    points = []
    colours, markers = {}, {}
    for e, row in rows_by_experiment:
        x, y = row["reads_per_iter"], row["iact_log_post"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        marker = "*" if e.proposal == "MH" else SS_MARKERS.get(round(1 / e.ss_prop), "X")
        ax.scatter([x], [y], s=60 if marker == "*" else 34, marker=marker, color=e.colour,
                   edgecolor="white", linewidth=0.8, zorder=3)
        points.append((x, y))
        colours.setdefault(e.label, e.colour)
        if e.proposal != "MH":
            markers.setdefault(e.ss_label, marker)
    if not points:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xx = np.geomspace(x0, x1, 50)
    for c in 10.0 ** np.arange(np.floor(np.log10(x0 * y0)), np.ceil(np.log10(x1 * y1)) + 1):
        ax.plot(xx, c / xx, color=GRID_INK, linewidth=0.8, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    handles = {label: Line2D([], [], color=c, marker="o", linestyle="", markersize=6)
               for label, c in colours.items()}
    for label, m in sorted(markers.items(), key=lambda kv: int(kv[0].split("/")[1])):
        handle = Line2D([], [], color=MUTED_INK, marker=m, linestyle="", markersize=6)
        handles[f"$\\omega = {label}$"] = handle
    ax.set_xlabel("Row reads per iteration")
    ax.set_ylabel(r"IACT of $\log\pi$ (iterations)")
    fig.tight_layout(rect=(0, 0, 0.72, 1))
    fig.legend(handles.values(), handles.keys(), loc="center right", frameon=False)
    if SHOW_TITLES:
        fig.suptitle(f"{dataset}: lower left is better; grey lines join equal row reads "
                     f"per independent state", size=8, x=0.36)
    fig.savefig(path)
    plt.close(fig)


def plot_smc_ess(plt, path, dataset, experiments, window=20):
    """Median over runs of ESS / N against step, as a running mean over
    `window` steps, one panel per subsample fraction."""
    panels = panels_by_ss(experiments)
    mh = [e for e in experiments if e.proposal == "MH"]
    if not panels:
        return
    fig, axes, cols = _panel_axes(plt, len(panels))
    kernel = np.ones(window) / window
    for ax, (ss, group) in zip(axes, panels.items()):
        for e in mh + group:
            frac = np.median([r["ess"] / r["particles"] for r in e.runs], axis=0)
            smooth = np.convolve(frac, kernel, mode="valid")
            ax.plot(np.arange(len(smooth)) + window, smooth, color=e.colour, label=e.label)
        ax.set_ylim(0, 1)
        _ss_title(ax, ss)
    for i, ax in enumerate(axes):
        if i % cols == 0:
            ax.set_ylabel(f"ESS / N ({window}-step mean)")
        if i >= len(axes) - cols:
            ax.set_xlabel("Step")
    _finish(plt, fig, axes, cols, path,
            f"{dataset}: SMC effective sample size before resampling, median over 16 runs")


def plot_smc_vs_mcmc(plt, path, dataset, mcmc, smc, reference):
    """Log posterior against row reads per run: an MCMC chain (solid) and a
    whole SMC run of N particles (dashed, the weighted mean), at equal cost."""
    from matplotlib.lines import Line2D

    panels = panels_by_ss(mcmc)
    smc_panels = panels_by_ss(smc)
    smc_mh = [e for e in smc if e.proposal == "MH"]
    if not panels:
        return
    grid = read_grid([e for es in panels.values() for e in es] + [reference] + smc)
    fig, axes, cols = _panel_axes(plt, len(panels))
    colours = {}
    for ax, (ss, group) in zip(axes, panels.items()):
        for e in [reference] + group:
            med, _, _ = curve_on_grid(e.runs, "rec_reads", "rec_lp", grid)
            ax.plot(grid, med, "-", color=e.colour, linewidth=1.2)
            colours.setdefault(e.label.replace(" (reference)", ""), e.colour)
        for e in smc_mh + smc_panels.get(ss, []):
            med, _, _ = curve_on_grid(e.runs, "rec_reads", "rec_wlp", grid)
            ax.plot(grid, med, "--", color=e.colour, linewidth=1.6)
            colours.setdefault(e.label, e.colour)
        ax.set_xscale("log")
        _ss_title(ax, ss)
    for i, ax in enumerate(axes):
        if i % cols == 0:
            ax.set_ylabel(r"$\log\pi$")
        if i >= len(axes) - cols:
            ax.set_xlabel("Row reads per chain or SMC run")
    handles = {label: Line2D([], [], color=c) for label, c in colours.items()}
    handles["MCMC chain"] = Line2D([], [], color=MUTED_INK, linestyle="-")
    handles[f"SMC run, {smc[0].runs[0]['particles']} particles"] = Line2D(
        [], [], color=MUTED_INK, linestyle="--")
    _finish(plt, fig, axes, cols, path,
            f"{dataset}: median log posterior of MCMC chains against SMC runs' weighted mean, "
            f"at equal row reads", handles=handles)


# --------------------------------------------------------------------------- #
# driving it
# --------------------------------------------------------------------------- #

def analyse_dataset(dataset, experiments, out, plt):
    references = [e for e in experiments if e.reference and e.sampler == "mcmc"]
    if not references:
        # No reference run named: the runs' own MH chains are the reference,
        # as they are when every sampler is run to the same budget.
        references = [e for e in experiments if e.sampler == "mcmc" and e.proposal == "MH"]
        for e in references:
            e.reference = True
    if len(references) != 1:
        raise SystemExit(f"{dataset}: expected one reference MCMC experiment, found "
                         f"{len(references)}")
    reference = references[0]
    mcmc = [e for e in experiments if e.sampler == "mcmc" and not e.reference]
    smc = [e for e in experiments if e.sampler == "smc"]
    level, ref_means = reference_level(reference)
    tables, figures = os.path.join(out, "tables"), os.path.join(out, "figures")

    print(f"\n=== {dataset}")
    print(f"level (lowest final-quarter mean of the {len(ref_means)} reference chains): "
          f"{level:,.1f}; their final-quarter means span "
          f"{min(ref_means):,.1f} to {max(ref_means):,.1f}")
    with open(os.path.join(tables, f"{dataset}_reference.csv"), "w") as f:
        f.write("chain,total_reads,final_quarter_mean_log_post,final_log_post\n")
        for i, (r, m) in enumerate(zip(reference.runs, ref_means)):
            f.write(f"{i},{r['total_reads']},{m!r},{r['rec_lp'][-1]!r}\n")

    rows = [mcmc_row(e, level, reference) for e in mcmc]
    rows.append(mcmc_row(reference, level, None))
    write_table(os.path.join(tables, f"{dataset}_mcmc"), rows, MCMC_COLUMNS,
                f"MCMC on {dataset}, each sampler over its own run: median over 16 chains, "
                f"apart from $\\hat R$ and the second-half means (nodes, log loss, accuracy) "
                f"pooled over them. Burn-in is in row reads, to the level {level:,.0f}.")

    budgets = common_budgets(mcmc)
    budget_rows = [row for b in budgets for row in
                   (budget_row(e, b, level) for e in mcmc + [reference]
                    if e.proposal != "MH" or e.reference)
                   if row["chains_at_budget"] >= len(reference.runs) / 2]
    write_table(os.path.join(tables, f"{dataset}_mcmc_budget"), budget_rows, BUDGET_COLUMNS,
                f"MCMC on {dataset} at equal budgets of row reads: medians over the chains "
                f"that reach the budget (a sampler is shown when at least half do); nodes and "
                f"prediction over the budget's second half, root splits over all of it. "
                f"Reached counts the chains at the level {level:,.0f} by the budget.")

    smc_rows = [smc_row(e, reference) for e in smc]
    write_table(os.path.join(tables, f"{dataset}_smc"), smc_rows, SMC_COLUMNS,
                f"SMC on {dataset}: medians over 16 runs; final values are the weighted "
                f"population's at the last stored step.")

    move_rows = [m for m in (moves_row(e) for e in mcmc + [reference] + smc) if m is not None]
    write_table(os.path.join(tables, f"{dataset}_moves"), move_rows, MOVES_COLUMNS,
                f"What became of the moves proposed on {dataset}, pooled over runs, as shares "
                f"of the moves made (stays excluded). SMC never rejects, so its accepted moves "
                f"are those that survived the proposal.")

    ref_loss = None
    if "test_log_loss" in reference.metrics:
        ref_loss = float(np.concatenate([m[second_half(r)] for r, m in
                                         zip(reference.runs, reference.metrics["test_log_loss"])]).mean())
    plot_progress(plt, os.path.join(figures, f"{dataset}_mcmc_log_post.pdf"), dataset, mcmc,
                  reference, "rec_lp", r"$\log\pi$", level=level, ref_values=ref_means)
    plot_progress(plt, os.path.join(figures, f"{dataset}_mcmc_log_loss.pdf"), dataset, mcmc,
                  reference, None, "Test log loss", metrics_key="test_log_loss", ref_value=ref_loss)
    plot_mixing(plt, os.path.join(figures, f"{dataset}_mcmc_mixing.pdf"), dataset,
                list(zip(mcmc, rows)))
    if smc:
        plot_smc_ess(plt, os.path.join(figures, f"{dataset}_smc_ess.pdf"), dataset, smc)
        plot_smc_vs_mcmc(plt, os.path.join(figures, f"{dataset}_smc_vs_mcmc.pdf"), dataset,
                         mcmc, smc, reference)

    _print_rows("MCMC, own runs", rows, ["sampler", "ss", "total_reads", "burnin_reads",
                                         "burnin_reached", "final_log_post",
                                         "reads_per_indep_log_post", "rhat_log_post",
                                         "nodes", "log_loss"])
    _print_rows("MCMC, equal budgets", budget_rows, ["budget", "sampler", "ss", "chains_at_budget",
                                                     "log_post", "reached", "nodes", "log_loss",
                                                     "root_features"])
    _print_rows("SMC", smc_rows, ["sampler", "ss", "total_reads", "start_ess", "median_ess_frac",
                                  "resample_rate", "final_log_post", "final_distinct",
                                  "final_nodes", "log_loss"])


def _print_rows(title, rows, keys):
    print(f"--- {title}")
    print("  ".join(f"{k[:14]:>14}" for k in keys))
    for row in rows:
        cells = []
        for k in keys:
            v = row.get(k, np.nan)
            cells.append(f"{v:>14}" if isinstance(v, str) else f"{v:>14.4g}")
        print("  ".join(cells))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="Results")
    p.add_argument("--run-ids", nargs="+", required=True,
                   help="the runs to compare, e.g. ch34 hints_2x4")
    p.add_argument("--reference", default=None,
                   help="a run holding longer MH reference chains (default: "
                        "the MH chains of --run-ids, as for runs to equal budgets)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="only these datasets (default: every one the runs hold)")
    p.add_argument("--out", default=None, help="default: <results-root>/analysis")
    p.add_argument("--jobs", type=int, default=min(available_cpus(), 8),
                   help="files read in parallel when building the cache")
    p.add_argument("--no-titles", action="store_true",
                   help="leave the titles off the figures, for a document that captions them")
    p.add_argument("--refresh", action="store_true",
                   help="re-read every .h5 instead of using the cache")
    args = p.parse_args()

    global SHOW_TITLES
    SHOW_TITLES = not args.no_titles
    out = args.out or os.path.join(args.results_root, "analysis")
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    experiments = discover(args.results_root, args.run_ids, args.reference)
    if args.datasets:
        experiments = [e for e in experiments if e.dataset in args.datasets]
    if not experiments:
        raise SystemExit("nothing to analyse")

    started = time.perf_counter()
    jobs = [(e, os.path.join(out, "cache"), args.refresh) for e in experiments]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            summaries = list(pool.map(summarise_file, jobs))
    else:
        summaries = [summarise_file(job) for job in jobs]
    for e, runs in zip(experiments, summaries):
        e.runs = runs
        e.metrics = load_metrics(e)
    print(f"read {len(experiments)} experiments in {time.perf_counter() - started:.1f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    for dataset in dict.fromkeys(e.dataset for e in experiments):
        analyse_dataset(dataset, [e for e in experiments if e.dataset == dataset], out, plt)
    print(f"\ntables and figures in {out}/")


if __name__ == "__main__":
    main()
