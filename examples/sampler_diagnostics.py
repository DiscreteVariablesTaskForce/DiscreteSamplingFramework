"""
Per-iteration diagnostics for the incremental decision tree samplers.

Runs MCMC and/or SMC under each of the three proposals and records, for every
iteration, what the sampler did and how good the tree it landed on was, so the
samplers and the proposal mechanisms can be put side by side on the same run.

    python examples/sampler_diagnostics.py --dataset wine
    python examples/sampler_diagnostics.py --dataset wine --run-id wine_baseline
    python examples/sampler_diagnostics.py --dataset covtype --iters 20000 \
        --metric-every 50 --proposals MH DA HINTS

Results land in Results/<run-id>/ (--run-id defaults to a timestamp, so two
sweeps never collide; name it explicitly to find a sweep again later, e.g.
`python examples/plot_diagnostics.py --run-id wine_baseline`). Inside that
directory sits a config.json recording what was asked for, and one .h5 per
(dataset, sampler, proposal) experiment -- see results_io.py for the layout.

What comes out, per configuration, as one .h5, one group per chain/run:

  per iteration (MCMC)      n_nodes, n_leaves, accepted
  per step (SMC)            ess, resampled, n_nodes/n_leaves mean and max
  per recorded iteration    accuracy, balanced_accuracy, macro precision /
                            recall / F1, log loss, Brier and the confusion
                            matrix, on both the training and the test split
  per move considered       move type, node, rows under it, the rows of the
                            subsample it was screened on, the surrogate ratio
                            that screen used, and what became of it
  per sample() call         the fate of the call as a whole

The move log is the one that makes the proposals comparable. MH considers one
move per iteration, DA considers one and screens it, HINTS considers one per
block -- so counting accepted *iterations* flatters HINTS, which does several
moves' worth of work inside one. The rows are tagged with their call index and
the per-block subsample size, so cost and yield can be read per move instead.

Costs and how they are kept down:

  * The metrics dominate everything else on a large dataset -- they route every
    row of both splits through the tree. --metric-every thins them; the
    iteration-level columns are still recorded every iteration.
  * Within that, a metric is recomputed only when the chain is on a different
    state object. A rejection leaves the state untouched, as does a "stay",
    so the previous value stands.
  * The move log is off unless asked for (--no-moves turns it off) and costs a
    few list appends per move when on. It does not touch the RNG, so a run with
    it on is the same run.

MCMC chains are independent and run in parallel across processes.
"""
import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import logsumexp
from sklearn import datasets
from sklearn.model_selection import train_test_split

from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
from discretesampling.domain import incremental_decision_tree as idt
from discretesampling.domain.incremental_decision_tree import diagnostics as dg
from results_io import resolve_run_id, save_experiment_hdf5, write_run_config


# --------------------------------------------------------------------------- #
# defaults, shared with run_experiments.py
# --------------------------------------------------------------------------- #

DATASETS = ("wine", "digits", "covtype")

# One experiment is this dict, with whatever it overrides. run_experiments.py
# lists several of these; main() builds one straight from its CLI arguments.
DEFAULT_CFG = dict(
    dataset="wine",
    samplers=["mcmc", "smc"],
    proposals=["MH", "DA", "HINTS"],
    chains=4,
    iters=5_000,
    steps=20,
    particles=200,
    metric_every=10,
    record_moves=True,
    ss_prop=0.1,
    min_data=20,
    lam=15.0,
    min_samples_leaf=3,
    max_tree_size=10,
    jobs=min(8, (os.cpu_count() or 2) - 1),
)


# --------------------------------------------------------------------------- #
# problem set-up
# --------------------------------------------------------------------------- #

def load_data(name):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}: expected one of {DATASETS}")
    if name == "covtype":
        import pandas as pd
        X, y = datasets.fetch_covtype(return_X_y=True)
        y = pd.Categorical(y).codes
    elif name == "wine":
        X, y = datasets.load_wine(return_X_y=True)
    else:
        X, y = datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=0)
    # The problem rejects a feature it cannot draw a threshold inside, and
    # digits has corner pixels that are constant on the training split.
    keep = X_train.min(axis=0) < X_train.max(axis=0)
    return X_train[:, keep], X_test[:, keep], y_train, y_test


def build(cfg, X_train, y_train):
    """The problem, target and proposal for one configuration."""
    problem = idt.IncrementalTreeProblem(
        X_train, y_train, lam=cfg['lam'],
        min_samples_leaf=cfg['min_samples_leaf'],
        max_tree_size=cfg['max_tree_size'])
    target = idt.IncrementalTreeTarget(problem)
    name = cfg['proposal']
    if name == "MH":
        proposal = idt.IncrementalTreeProposal()
    elif name == "DA":
        proposal = idt.DAProposal(target, ss_prop=cfg['ss_prop'],
                                  min_data=cfg['min_data'])
    else:
        proposal = idt.HINTSProposal(target, ss_prop=cfg['ss_prop'],
                                     min_data=cfg['min_data'])
    proposal.record_moves = cfg['record_moves']
    return problem, target, proposal


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #

class Recorder:
    """
    Collects per-iteration columns, and the predictive metrics on a thinned
    subset of iterations.

    The metric cache is keyed on the identity of the state object, not on the
    accept flag: a rejected step and an accepted "stay" both hand back the very
    same object, and only the first of those is visible as a rejection.
    """

    def __init__(self, splits, num_classes, every):
        self.splits = splits            # [(prefix, X, y), ...]
        self.num_classes = num_classes
        self.every = every
        self.cols = defaultdict(list)
        self._last_key = object()
        self._last = None

    def add(self, **kv):
        for k, v in kv.items():
            self.cols[k].append(v)

    def metrics(self, i, states, key, weights=None):
        """Record the predictive metrics for `states`, if this iteration is one
        of the thinned ones. `key` identifies the state set for the cache."""
        if i % self.every:
            return
        if key is not self._last_key:
            out = {}
            for prefix, X, y in self.splits:
                out.update(idt.evaluate(states, X, y, weights=weights,
                                        num_classes=self.num_classes,
                                        prefix=prefix))
            self._last, self._last_key = out, key
        self.cols['metric_iter'].append(i)
        for k, v in self._last.items():
            self.cols[k].append(v)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.cols.items()}


def counters_of(proposal, target, prefix=""):
    """The run-level counters, flattened for an .npz."""
    out = {prefix + "prop_" + k: v for k, v in proposal.counters().items()}
    out.update({prefix + "target_" + k: v for k, v in target.counters().items()})
    return out


# --------------------------------------------------------------------------- #
# the two samplers
# --------------------------------------------------------------------------- #

def run_mcmc(cfg):
    X_train, X_test, y_train, y_test = load_data(cfg['dataset'])
    problem, target, proposal = build(cfg, X_train, y_train)
    mcmc = DiscreteVariableMCMC(idt.IncrementalTree, target,
                                idt.IncrementalTreeInitialProposal(problem),
                                proposal=proposal)

    rec = Recorder([("train_", X_train, y_train), ("test_", X_test, y_test)],
                   problem.num_classes, cfg['metric_every'])

    def record(i, current, accepted):
        rec.add(n_nodes=len(current.tree), n_leaves=len(current.leaf_idx),
                accepted=accepted)
        rec.metrics(i, [current], current)

    mcmc.sample(cfg['iters'], seed=cfg['seed'], verbose=False, callback=record)

    out = rec.arrays()
    out.update(counters_of(proposal, target))
    out['acceptance_rate'] = mcmc.acceptance_rate
    out['tail_acceptance_rate'] = mcmc.tail_acceptance_rate
    if cfg['record_moves']:
        out.update(proposal.move_log.arrays())
    return out


def run_smc(cfg):
    X_train, X_test, y_train, y_test = load_data(cfg['dataset'])
    problem, target, proposal = build(cfg, X_train, y_train)
    smc = DiscreteVariableSMC(idt.IncrementalTree, target,
                              idt.IncrementalTreeInitialProposal(problem),
                              proposal=proposal, Lkernel=proposal.lkernel())

    rec = Recorder([("train_", X_train, y_train), ("test_", X_test, y_test)],
                   problem.num_classes, cfg['metric_every'])

    def record(t, particles, logWeights, neff, resampled):
        sizes = np.fromiter((len(p.tree) for p in particles), np.int64,
                            len(particles))
        leaves = np.fromiter((len(p.leaf_idx) for p in particles), np.int64,
                             len(particles))
        rec.add(ess=neff, resampled=resampled,
                mean_nodes=sizes.mean(), max_nodes=sizes.max(),
                mean_leaves=leaves.mean(),
                # Where in the move log this step's calls end, so the moves can
                # be sliced back out per step: the proposal is shared by every
                # particle, so its call index runs over particle-steps.
                calls_at_step=proposal.n_calls)
        # The ensemble estimator is the weighted one; an unweighted read of the
        # particles is not what SMC is targeting. The weights are unnormalised
        # here, so normalise before exponentiating.
        w = np.exp(logWeights - logsumexp(logWeights))
        rec.metrics(t, particles, particles, weights=w)

    particles = smc.sample(cfg['steps'], cfg['particles'], seed=cfg['seed'],
                           verbose=False, callback=record)

    out = rec.arrays()
    out.update(counters_of(proposal, target))
    out['ess_history'] = np.asarray(smc.ess_history)
    out['resampled_history'] = np.asarray(smc.resampled_history)
    out['final_logWeights'] = smc.logWeights
    final = idt.evaluate(particles, X_test, y_test,
                         weights=np.exp(smc.logWeights),
                         num_classes=problem.num_classes, prefix="final_test_")
    out.update({k: v for k, v in final.items()})
    if cfg['record_moves']:
        out.update(proposal.move_log.arrays())
    return out


def run(cfg):
    return run_mcmc(cfg) if cfg['sampler'] == "mcmc" else run_smc(cfg)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def report(cfg, results):
    """One block per configuration, averaged over the runs of it."""
    tag = f"{cfg['sampler'].upper()}-{cfg['proposal']}"
    print(f"\n{'=' * 78}\n{tag}   ({len(results)} run(s))\n{'=' * 78}")

    def mean_of(key):
        vals = [r[key] for r in results if key in r]
        return float(np.mean(vals)) if vals else float('nan')

    if cfg['sampler'] == "mcmc":
        print("acceptance rate      %.4f   (last 1000: %.4f)"
              % (mean_of('acceptance_rate'), mean_of('tail_acceptance_rate')))
        sizes = np.concatenate([r['n_nodes'] for r in results])
        print("tree size            mean %.2f  median %d  max %d"
              % (sizes.mean(), int(np.median(sizes)), sizes.max()))
    else:
        ess = np.vstack([r['ess_history'] for r in results])
        print("ESS                  first %.1f  min %.1f  final %.1f  (of %d particles)"
              % (ess[:, 0].mean(), ess.min(axis=1).mean(), ess[:, -1].mean(),
                 cfg['particles']))
        resampled = np.vstack([r['resampled_history'] for r in results])
        print("resampled            %d of %d steps"
              % (resampled.sum(axis=1).mean(), resampled.shape[1]))

    print("proposal time        %.2f s over %d calls"
          % (mean_of('prop_sample_time'), mean_of('prop_calls')))
    print("target evals         %d full, %d subset, %d memo hits"
          % (mean_of('target_full_evals'), mean_of('target_subset_evals'),
             mean_of('target_memo_hits')))

    for split in ("train_", "test_"):
        if split + "accuracy" not in results[0]:
            continue
        acc = np.vstack([r[split + 'accuracy'] for r in results])
        f1 = np.vstack([r[split + 'macro_f1'] for r in results])
        ll = np.vstack([r[split + 'log_loss'] for r in results])
        print("%-20s final accuracy %.4f  macro F1 %.4f  log loss %.4f"
              % (split.rstrip('_'), acc[:, -1].mean(), f1[:, -1].mean(),
                 ll[:, -1].mean()))

    if cfg['record_moves'] and 'move' in results[0]:
        log = dg.MoveLog()
        for r in results:
            log.move.extend(r['move'].tolist())
            log.outcome.extend(r['outcome'].tolist())
        print("\nmoves considered, by type and fate:")
        print(log.format_counts())
        rows = np.concatenate([r['rows'] for r in results])
        subset = np.concatenate([r['subset'] for r in results])
        screened = subset > 0
        if screened.any():
            print("screened on %.1f%% of the rows they act on, on average"
                  % (100.0 * (subset[screened] / np.maximum(rows[screened], 1)).mean()))


# --------------------------------------------------------------------------- #
# running one experiment (samplers x proposals x chains) into a results dir
# --------------------------------------------------------------------------- #

# The sweep dimensions: how many configs to build and how to run them, rather
# than a parameter of any one of them. Kept out of the per-chain cfg dict `run`
# is called with.
_SWEEP_KEYS = ("samplers", "proposals", "chains", "jobs")


def run_sweep(cfg, results_dir, name=None, jobs=None):
    """
    Run one experiment's samplers x proposals x chains cross product and write
    <name>_<sampler>_<proposal>.h5 into results_dir for each (sampler,
    proposal) pair. This is main()'s body, factored out so a driver script can
    run a list of experiments into one results directory -- see
    run_experiments.py and experiments.py.

    `cfg` is overlaid on DEFAULT_CFG, so any key it does not set keeps its
    default. `name` defaults to cfg['dataset']; give experiments that share a
    dataset distinct names, or run_sweep would happily overwrite one's .h5
    files with the other's.
    """
    cfg = dict(DEFAULT_CFG, **cfg)
    jobs = cfg['jobs'] if jobs is None else jobs
    name = name or cfg['dataset']

    base = {k: v for k, v in cfg.items() if k not in _SWEEP_KEYS}
    configs = [dict(base, sampler=s, proposal=q, seed=seed)
               for s in cfg['samplers'] for q in cfg['proposals']
               for seed in range(cfg['chains'])]

    if jobs > 1 and len(configs) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(run, configs, chunksize=1))
    else:
        results = [run(c) for c in configs]

    grouped = defaultdict(list)
    for c, res in zip(configs, results):
        grouped[(c['sampler'], c['proposal'])].append((c, res))

    written = []
    for (sampler, proposal), pairs in grouped.items():
        cfgs, runs = zip(*pairs)
        path = os.path.join(results_dir, f"{name}_{sampler}_{proposal}.h5")
        # One .h5 per configuration, one group per run, so runs of unequal
        # length (different numbers of accepted moves) stay separable.
        save_experiment_hdf5(path, list(runs))
        report(cfgs[0], list(runs))
        print(f"\nwritten to {path}")
        written.append(path)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, default=DEFAULT_CFG['dataset'])
    p.add_argument("--samplers", nargs="+", choices=["mcmc", "smc"],
                   default=DEFAULT_CFG['samplers'])
    p.add_argument("--proposals", nargs="+", choices=["MH", "DA", "HINTS"],
                   default=DEFAULT_CFG['proposals'])
    p.add_argument("--chains", type=int, default=DEFAULT_CFG['chains'],
                   help="independent MCMC chains / SMC runs")
    p.add_argument("--iters", type=int, default=DEFAULT_CFG['iters'], help="MCMC iterations")
    p.add_argument("--steps", type=int, default=DEFAULT_CFG['steps'], help="SMC steps")
    p.add_argument("--particles", type=int, default=DEFAULT_CFG['particles'])
    p.add_argument("--metric-every", type=int, default=DEFAULT_CFG['metric_every'],
                   help="record the predictive metrics every k iterations")
    p.add_argument("--no-moves", action="store_true", help="skip the per-move log")
    p.add_argument("--ss-prop", type=float, default=DEFAULT_CFG['ss_prop'])
    p.add_argument("--min-data", type=int, default=DEFAULT_CFG['min_data'])
    p.add_argument("--lam", type=float, default=DEFAULT_CFG['lam'])
    p.add_argument("--min-samples-leaf", type=int, default=DEFAULT_CFG['min_samples_leaf'])
    p.add_argument("--max-tree-size", type=int, default=DEFAULT_CFG['max_tree_size'])
    p.add_argument("--jobs", type=int, default=DEFAULT_CFG['jobs'])
    p.add_argument("--results-root", default="Results")
    p.add_argument("--run-id", default=None,
                   help="results directory name under --results-root "
                        "(default: a timestamp)")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG,
               dataset=args.dataset, samplers=args.samplers, proposals=args.proposals,
               chains=args.chains, iters=args.iters, steps=args.steps,
               particles=args.particles, metric_every=args.metric_every,
               record_moves=not args.no_moves, ss_prop=args.ss_prop,
               min_data=args.min_data, lam=args.lam,
               min_samples_leaf=args.min_samples_leaf, max_tree_size=args.max_tree_size,
               jobs=args.jobs)

    run_id = resolve_run_id(args.run_id)
    results_dir = os.path.join(args.results_root, run_id)
    os.makedirs(results_dir, exist_ok=True)
    write_run_config(results_dir, dict(cfg, results_root=args.results_root, run_id=run_id))

    run_sweep(cfg, results_dir)

    print(f"\nResults in: {results_dir}"
          f"\n  python examples/plot_diagnostics.py --run-id {run_id} --name {args.dataset}")


if __name__ == "__main__":
    main()
