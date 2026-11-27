"""
Per-iteration diagnostics for the incremental decision tree samplers.

Runs MCMC and/or SMC under each of the three proposals and records, for every
iteration, what the sampler did and which tree it landed on, so the samplers
and the proposal mechanisms can be put side by side on the same run.

    python examples/incremental_decision_tree/sampler_diagnostics.py --dataset wine
    python examples/incremental_decision_tree/sampler_diagnostics.py --dataset wine --run-id wine_baseline
    python examples/incremental_decision_tree/sampler_diagnostics.py --dataset covtype --iters 20000 \
        --proposals MH DA FlatHINTS

Results land in Results/<run-id>/ (--run-id defaults to a timestamp, so two
sweeps never collide; name it explicitly to find a sweep again later, e.g.
`python examples/incremental_decision_tree/plot_diagnostics.py --run-id wine_baseline`). Inside that
directory sits a config.json recording what was asked for, and one
<dataset>_<sampler>_<proposal>_<ss_prop * 100>[_init<N>|_init-<name>].h5 per configuration -- see
results_io.py for the layout.

Sampling and evaluation are separate steps
------------------------------------------
This script samples and stores states. It computes no predictive metrics at
all. To get accuracy, F1, log loss and the rest:

    python examples/incremental_decision_tree/sampler_diagnostics.py --dataset wine --run-id wine_baseline
    python examples/incremental_decision_tree/evaluate_results.py --run-id wine_baseline
    python examples/incremental_decision_tree/plot_diagnostics.py --run-id wine_baseline

The split is what makes a long run affordable and re-readable. One metric
evaluation routes every row of both splits through a tree, which on covtype
costs an order of magnitude more than the sampler iteration that produced the
tree; evaluating inline therefore made the diagnostics *be* the run, and fixed
at sampling time which metrics -- on which splits, at which thinning -- you
would ever be able to look at. Storing the states instead means the sampler
runs at sampler speed, and any metric can be computed, re-computed and
re-thinned afterwards from the same file.

What comes out, per configuration, as one .h5, one group per chain/run:

  per iteration (MCMC)      n_nodes, n_leaves, accepted
  per step (SMC)            ess, resampled, n_nodes/n_leaves mean and max
  per iteration/step        iter_time and cumulative_time, in seconds, for
                            plotting a metric against wallclock cost rather
                            than against iteration count. The clock excludes
                            the state recording itself, so it measures the
                            sampler and not the diagnostics.
  per iteration/step        row_reads, the cumulative rows the sampler has read
                            (idt.RowReads): the cost measure that does not
                            depend on how fast the code is. Drawing the
                            starting trees is not counted.
  per stored record         the tree at that iteration (MCMC), or every
                            particle of that step plus its log weight (SMC)
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

  * Only *distinct* states are stored, and identity settles which those are: a
    rejected move hands back the object the chain already held, and resampling
    puts one particle object in many slots. What a record costs is one integer
    per particle. See states.py -- the compression is exact, not a thinning.
  * --store-every thins the records themselves, for a run long enough that even
    the integers matter. The iteration-level columns are still recorded every
    iteration. --no-states turns state recording off entirely, leaving a run
    that can be timed but not evaluated.
  * The move log is off unless asked for (--no-moves turns it off) and costs a
    few list appends per move when on. It does not touch the RNG, so a run with
    it on is the same run.

MCMC chains are independent and run in parallel across processes.
"""
import math
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
from discretesampling.domain import incremental_decision_tree as idt
from discretesampling.domain.incremental_decision_tree import diagnostics as dg
from discretesampling.domain.incremental_decision_tree.states import StateRecorder
from results_io import (INIT_NAME, PRIOR_INIT, PROPOSALS, available_cpus, experiment_filename,
                        experiment_key, experiment_name, init_of,
                        resolve_run_id, save_experiment_hdf5, write_run_config)


# --------------------------------------------------------------------------- #
# defaults, shared with run_experiments.py
# --------------------------------------------------------------------------- #

DATASETS = ("wine", "digits", "covtype", "mnist", "hepmass", "susy", "frogs", "sensorless")

# Proposal arguments an experiment may set, by the proposal that takes them.
# They are deliberately absent from DEFAULT_CFG: an experiment that names one
# gets it, and one that does not gets whatever the proposal's own signature
# says, rather than a second opinion kept here.
PROPOSAL_KNOBS = {
    "HINTS": ("levels", "branching", "hierarchy", "equal_blocks", "temper", "bucket_rows",
              "lazy_counts", "fused_buckets"),
    "FlatHINTS": ("equal_blocks",),
}
ALL_KNOBS = frozenset(k for knobs in PROPOSAL_KNOBS.values() for k in knobs)

# One experiment is this dict, with whatever it overrides. run_experiments.py
# lists several of these; main() builds one straight from its CLI arguments.
DEFAULT_CFG = dict(
    dataset="wine",
    samplers=["mcmc", "smc"],
    proposals=["MH", "DA", "FlatHINTS"],
    chains=4,
    # The seed of the first chain: chains run seeds chain_offset, ...,
    # chain_offset + chains - 1, so that one experiment's chains can be split
    # over several jobs (run_experiments.py --split) and still be the chains
    # an unsplit run would have been, chain i starting from the same tree.
    chain_offset=0,
    iters=5_000,
    steps=20,
    particles=200,
    store_every=1,
    # A budget of row reads (problem.row_reads) per chain or SMC run, the
    # cost measure the samplers are compared on. Set, a run stops at the first
    # iteration or step that reaches it, as every run of Chapter 2 stops at a
    # budget of full evaluations; `iters` and `steps` are then only upper
    # limits, and the states are stored at `records` evenly spaced row-read
    # marks instead of every `store_every` iterations, so that every sampler
    # is recorded at the same costs. None runs for `iters`/`steps`.
    budget=None,
    records=5_000,
    record_states=True,
    record_moves=True,
    ss_prop=0.1,
    min_data=20,
    # The exponent gamma of every proposal's subtree-root draw: the root is
    # drawn with probability proportional to (rows under it + 1)^-gamma. 0 is
    # the uniform draw of the thesis runs (see moves.draw_weighted_subtree).
    root_weight=0.0,
    lam=15.0,
    min_samples_leaf=3,
    max_tree_size=None,
    # Where the runs start. None draws every chain's and every particle's
    # starting tree from the prior, conditioned split by split on being
    # admissible (files tagged _init-prior). Otherwise every chain, every
    # particle, both samplers and every proposal start from one tree: 0 a
    # stump (untagged files), N > 0 one random tree of N internal nodes
    # (_init<N>). See IncrementalTreeInitialProposal.
    init_tree_size=None,
    # Or the starting tree itself, in place of a random one: a list of
    # [node, left, right, feat, thr] rows, one per internal node, the root
    # being node 0 (see idt.tree_rows; any tree's own .tree is such a list).
    # feat numbers the columns load_data hands back, i.e. after it has dropped
    # the constant ones. init_tree_name is what the result files are tagged
    # with (letters, digits, '-'); left out, a digest of the rows is used.
    init_tree=None,
    init_tree_name=None,
    jobs=min(8, max(1, available_cpus() - 1)),
)


# --------------------------------------------------------------------------- #
# problem set-up
# --------------------------------------------------------------------------- #

def load_frogs():
    df = pd.read_csv('Datasets/Frogs_MFCCs.csv')
    # remove Family and Genus and Target columns, leaving Species as the target
    df = df.drop(columns=['Family', 'Genus', 'Target'])
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    return X, pd.Categorical(y).codes

def load_susy():
    df = pd.read_csv('Datasets/SUSY.csv')
    # first column is the target, the rest are the features, there are no headers
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').to_numpy()
    return X, pd.Categorical(y).codes

def load_sensorless():
    """Sensorless drive diagnosis: 58,509 rows of 48 continuous features of
    the motor current, and 11 classes of fault, 1 to 11 in the last column."""
    df = pd.read_csv('Datasets/Sensorless_drive_diagnosis.csv', header=None)
    return df.iloc[:, :-1].to_numpy(dtype=np.float64), pd.Categorical(df.iloc[:, -1]).codes


def load_hepmass():
    df1 = pd.read_csv('Datasets/hepmass_train.csv')
    df2 = pd.read_csv('Datasets/hepmass_test.csv')
    df1.columns = range(df1.shape[1])
    df2.columns = range(df2.shape[1])
    df = pd.concat([df1, df2], ignore_index=True)
    # first column is the label, the rest are the features except the last column
    y = df.iloc[:, 0].values
    # the last column is mass, which we will drop
    X = df.iloc[:, 1:-1].values
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
    return X.to_numpy(), pd.Categorical(y).codes

def load_mnist():
    df1 = pd.read_csv('Datasets/mnist_train.csv')
    df2 = pd.read_csv('Datasets/mnist_test.csv')
    # reset column headings of df2 and df1 to be counting from 0
    df1.columns = range(df1.shape[1])
    df2.columns = range(df2.shape[1])
    df = pd.concat([df1, df2], ignore_index=True)
    # first column is the label, the rest are the pixels
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
    X = X.loc[:, (X != 0).any(axis=0)]
    # drop all columns with <10 non zero values
    X = X.loc[:, (X != 0).sum(axis=0) >= 10]
    return X.to_numpy(), pd.Categorical(y).codes


# A directory of datasets already split, one <name>.npz each, written by
# prepare_data. When this variable names one, load_data reads from it and
# nothing else: the SLURM jobs set it, so the CSV is parsed once at submission
# rather than by every chain, and no compute node needs the network for a
# scikit-learn download. Unset, load_data splits the data itself, as before.
DATA_CACHE_ENV = "DSF_DATA_CACHE"


def prepared_path(name, cache_dir):
    return os.path.join(cache_dir, f"{name}.npz")


def prepare_data(name, cache_dir):
    """Split dataset `name` as load_data does and write it to cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = split_data(name)
    path = prepared_path(name, cache_dir)
    tmp = f"{path}.{os.getpid()}.tmp.npz"       # np.savez wants the suffix
    np.savez(tmp, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    os.replace(tmp, path)
    return path


def load_data(name):
    """(X_train, X_test, y_train, y_test): the stratified 80/20 split, with
    constant training columns dropped. Read from the prepared copy when
    DSF_DATA_CACHE is set, and split here otherwise."""
    cache_dir = os.environ.get(DATA_CACHE_ENV)
    if not cache_dir:
        return split_data(name)
    path = prepared_path(name, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{DATA_CACHE_ENV}={cache_dir} has no {name}.npz: prepare it with "
            f"run_experiments.py --prepare (slurm/submit.sh does)")
    with np.load(path) as f:
        return f["X_train"], f["X_test"], f["y_train"], f["y_test"]


def split_data(name):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}: expected one of {DATASETS}")
    if name == "covtype":
        X, y = datasets.fetch_covtype(return_X_y=True)
        y = pd.Categorical(y).codes
    elif name == "wine":
        X, y = datasets.load_wine(return_X_y=True)
    elif name == "digits":
        X, y = datasets.load_digits(return_X_y=True)
    else:
        loader = dict(frogs=load_frogs, susy=load_susy, hepmass=load_hepmass,
                      mnist=load_mnist, sensorless=load_sensorless)[name]
        X, y = loader()

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
    elif name == "HINTS":
        # Passed through only when an experiment actually sets one, so the
        # proposal's own defaults stay the single place they are written down.
        # Restating them here is how a run silently ends up on a setting nobody
        # chose: a copy of a default here outlives a change to the real one.
        knobs = {k: cfg[k] for k in PROPOSAL_KNOBS[name] if k in cfg}
        proposal = idt.HINTSProposal(target, ss_prop=cfg['ss_prop'],
                                     min_data=cfg['min_data'], **knobs)
    elif name == "FlatHINTS":
        knobs = {k: cfg[k] for k in PROPOSAL_KNOBS[name] if k in cfg}
        proposal = idt.FlatHINTSProposal(target, ss_prop=cfg['ss_prop'],
                                         min_data=cfg['min_data'], **knobs)
    else:
        # Never a fallback: an unrecognised name used to land here and run as
        # FlatHINTS, so a run labelled one thing silently sampled with another.
        raise ValueError(f"unknown proposal {name!r}: expected one of {PROPOSALS}")
    proposal.record_moves = cfg['record_moves']
    if cfg.get('root_weight'):
        proposal.root_weight = float(cfg['root_weight'])
    return problem, target, proposal


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #

class Recorder:
    """
    Collects the per-iteration columns, and the states themselves on a thinned
    subset of iterations.

    A run set by iterations stores a state every `every` iterations and the
    columns every iteration. A run given a budget of row reads (`marks`, the
    row reads at which to record, from budget_marks) records once per mark
    instead, the state of the first iteration to reach it, so every sampler's
    records sit at the same costs whatever an iteration of it costs, and every
    chain that reaches its budget has exactly one record per mark. An
    iteration that passes several marks at once is recorded once for each:
    the repeats cost an index each, since only distinct states are stored. With `thin_columns`
    the per-iteration columns are kept at those records only, which is what
    keeps a run of millions of cheap iterations in memory; the run-level
    counts (iterations, full evaluations, accepted proposals) are kept in
    full either way.

    No metric is evaluated here, or anywhere else in this file: what is kept is
    the tree, and evaluate_results.py turns stored trees into metrics
    afterwards. See the module docstring for why.
    """

    def __init__(self, problem, every, record_states=True, marks=None,
                 thin_columns=False):
        self.every = every
        self.marks = None if marks is None else np.asarray(marks, dtype=np.float64)
        self.thin_columns = bool(thin_columns) and self.marks is not None
        self._next = 0
        self.cols = defaultdict(list)
        self.states = (StateRecorder(problem.num_classes, problem.alpha)
                       if record_states else None)
        self._mark = time.perf_counter()
        self._elapsed = 0.0

    def start(self):
        """Start the clock, so the timings measure sampling rather than the
        problem set-up that preceded it."""
        self._mark = time.perf_counter()
        self._elapsed = 0.0

    def tick(self):
        """
        This iteration's own duration, and the run's cumulative time, both in
        seconds, recorded every iteration unless the columns are thinned.

        The clock runs from the end of the last iteration's recording to the
        start of this one, so it measures the sampler alone: what this recorder
        spends is not charged to the algorithm. Storing a state is far cheaper
        than the metric evaluation it replaced, but it is still work the
        sampler did not ask for, and on a run where a chain rarely moves the
        recording would otherwise show up as a per-proposal cost difference
        that has nothing to do with the proposals.
        """
        now = time.perf_counter()
        iter_time = now - self._mark
        self._elapsed += iter_time
        if not self.thin_columns:
            self.cols['iter_time'].append(iter_time)
            self.cols['cumulative_time'].append(self._elapsed)

    def due(self, i, row_reads):
        """
        How many records iteration (or step) `i`, after which the run has read
        `row_reads` rows, makes: for a run set by iterations, one if it is an
        `every`-th iteration and none otherwise; for a run with a budget, one
        for each mark it is the first to reach. With thinned columns, the
        iteration and the cumulative time are recorded here, once per record.
        """
        if self.marks is None:
            return int(i % self.every == 0)
        reached = int(np.searchsorted(self.marks, row_reads, side='right'))
        count = max(reached - self._next, 0)
        self._next = max(reached, self._next)
        if self.thin_columns:
            self.cols['iteration'].extend([i] * count)
            self.cols['cumulative_time'].extend([self._elapsed] * count)
        return count

    def add(self, count=1, **kv):
        """Append each column's value, `count` times."""
        for k, v in kv.items():
            self.cols[k].extend([v] * count)

    def store(self, i, states, log_weights=None, due=1):
        """
        Store this iteration's ensemble -- the single current tree for MCMC,
        the whole particle set plus its log weights for SMC -- as `due`
        records (from due()), none if it is not one to record.
        """
        try:
            if self.states is None:
                return
            for _ in range(due):
                self.states.record(states, iteration=i, log_weights=log_weights)
        finally:
            # Restart the clock: whatever was just spent recording belongs to
            # the diagnostics, not to the next sampler iteration.
            self._mark = time.perf_counter()

    def arrays(self):
        out = {k: np.asarray(v) for k, v in self.cols.items()}
        if self.states is not None:
            out.update(self.states.arrays())
        return out


def budget_marks(cfg):
    """The row reads at which a run with a budget stores its states: `records`
    evenly spaced marks, the last of them the budget itself. None for a run
    set by iterations."""
    if cfg['budget'] is None:
        return None
    return np.linspace(cfg['budget'] / cfg['records'], cfg['budget'], cfg['records'])


def counters_of(proposal, target, prefix=""):
    """The run-level counters, flattened for an .npz."""
    out = {prefix + "prop_" + k: v for k, v in proposal.counters().items()}
    out.update({prefix + "target_" + k: v for k, v in target.counters().items()})
    return out


# --------------------------------------------------------------------------- #
# the two samplers
# --------------------------------------------------------------------------- #

def initial_proposal(cfg, problem):
    """Where every chain and particle of a configuration starts: the tree it
    was given, else a random one of init_tree_size nodes (a stump for 0),
    else a tree of its own drawn from the prior."""
    return idt.IncrementalTreeInitialProposal(
        problem, cfg['init_tree_size'], tree=cfg['init_tree'])


def run_mcmc(cfg):
    # Only the training split reaches the sampler now: the test split is what
    # the metrics are read on, and those happen in evaluate_results.py.
    X_train, _, y_train, _ = load_data(cfg['dataset'])
    problem, target, proposal = build(cfg, X_train, y_train)
    mcmc = DiscreteVariableMCMC(idt.IncrementalTree, target,
                                initial_proposal(cfg, problem),
                                proposal=proposal)

    budget = cfg['budget']
    rec = Recorder(problem, cfg['store_every'], cfg['record_states'],
                   marks=budget_marks(cfg), thin_columns=True)
    # What became of each iteration's proposal, counted over the whole run:
    # whether it reached the accept step on the full data (the proposal did
    # not stay put) and whether it was then accepted.
    tally = {'iterations': 0, 'full_evals': 0, 'accepted_full': 0}
    moved = [proposal.n_moved]

    def record(i, current, accepted):
        rec.tick()
        reads = problem.row_reads
        full = proposal.n_moved != moved[0]
        moved[0] = proposal.n_moved
        tally['iterations'] += 1
        tally['full_evals'] += full
        tally['accepted_full'] += full and accepted
        due = rec.due(i, reads)
        rec.add(due if rec.thin_columns else 1,
                n_nodes=len(current.tree), n_leaves=len(current.leaf_idx),
                accepted=accepted, row_reads=reads)
        rec.store(i, (current,), due=due)
        return budget is not None and reads >= budget

    rec.start()
    mcmc.sample(cfg['iters'], seed=cfg['seed'], verbose=False, callback=record)

    out = rec.arrays()
    out.update(counters_of(proposal, target))
    out.update(tally)
    out['acceptance_rate'] = mcmc.acceptance_rate
    out['tail_acceptance_rate'] = mcmc.tail_acceptance_rate
    if budget is not None:
        out['budget'] = float(budget)
    if cfg['record_moves']:
        out.update(proposal.move_log.arrays())
    return out


def run_smc(cfg):
    X_train, _, y_train, _ = load_data(cfg['dataset'])
    problem, target, proposal = build(cfg, X_train, y_train)
    smc = DiscreteVariableSMC(idt.IncrementalTree, target,
                              initial_proposal(cfg, problem),
                              proposal=proposal, Lkernel=proposal.lkernel())

    # SMC takes far fewer steps than MCMC takes iterations, and its per-step
    # columns (the ESS above all) are wanted at every step, so only its states
    # are thinned by the budget.
    budget = cfg['budget']
    rec = Recorder(problem, cfg['store_every'], cfg['record_states'],
                   marks=budget_marks(cfg))

    def record(t, particles, logWeights, neff, resampled):
        rec.tick()
        sizes = np.fromiter((len(p.tree) for p in particles), np.int64,
                            len(particles))
        leaves = np.fromiter((len(p.leaf_idx) for p in particles), np.int64,
                             len(particles))
        reads = problem.row_reads
        rec.add(ess=neff, resampled=resampled,
                mean_nodes=sizes.mean(), max_nodes=sizes.max(),
                mean_leaves=leaves.mean(),
                # Every particle's moves so far, as one running total.
                row_reads=reads,
                # Where in the move log this step's calls end, so the moves can
                # be sliced back out per step: the proposal is shared by every
                # particle, so its call index runs over particle-steps.
                calls_at_step=proposal.n_calls)
        # Every particle, plus the weight it carries. The SMC estimator is the
        # weighted one -- an unweighted read of the particles is not what SMC
        # targets -- so the weights are stored with the states rather than
        # thrown away; they are unnormalised here and StateSeries.weights
        # normalises on the way out.
        rec.store(t, particles, log_weights=logWeights, due=rec.due(t, reads))
        return budget is not None and reads >= budget

    rec.start()
    smc.sample(cfg['steps'], cfg['particles'], seed=cfg['seed'],
               verbose=False, callback=record)

    out = rec.arrays()
    out.update(counters_of(proposal, target))
    out['ess_history'] = np.asarray(smc.ess_history)
    out['resampled_history'] = np.asarray(smc.resampled_history)
    # The weights after the final normalise, which the last callback ran too
    # early to see: the state stored for the last step carries the weights as
    # they were before it.
    out['final_logWeights'] = smc.logWeights
    if budget is not None:
        out['budget'] = float(budget)
    if cfg['record_moves']:
        out.update(proposal.move_log.arrays())
    return out


SAMPLERS = ("mcmc", "smc")


def run(cfg):
    if cfg['sampler'] not in SAMPLERS:
        raise ValueError(f"unknown sampler {cfg['sampler']!r}: expected one of {SAMPLERS}")
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
        # Runs given a budget take different numbers of steps, so each run is
        # summarised on its own and the summaries averaged.
        ess = [np.asarray(r['ess_history']) for r in results]
        print("ESS                  first %.1f  min %.1f  final %.1f  (of %d particles)"
              % (np.mean([e[0] for e in ess]), np.mean([e.min() for e in ess]),
                 np.mean([e[-1] for e in ess]), cfg['particles']))
        resampled = [np.asarray(r['resampled_history']) for r in results]
        print("resampled            %d of %d steps"
              % (np.mean([r.sum() for r in resampled]),
                 np.mean([len(r) for r in resampled])))

    print("proposal time        %.2f s over %d calls"
          % (mean_of('prop_sample_time'), mean_of('prop_calls')))
    print("target evals         %d full, %d subset, %d memo hits"
          % (mean_of('target_full_evals'), mean_of('target_subset_evals'),
             mean_of('target_memo_hits')))

    if 'state_index' in results[0]:
        # Recorded against stored: how much the identity dedup actually saved,
        # which is the number to look at before reaching for --store-every.
        recorded = float(np.mean([r['state_index'].size for r in results]))
        stored = float(np.mean([len(r['state_tree_lengths']) for r in results]))
        print("states               %d stored of %d recorded  (%.1fx)"
              % (stored, recorded, recorded / max(stored, 1)))

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
_SWEEP_KEYS = ("samplers", "proposals", "chains", "chain_offset", "jobs")


def validate_cfg(cfg):
    """
    `cfg` (already overlaid on DEFAULT_CFG) with samplers and proposals as
    lists, or a ValueError naming what is wrong with it.

    Every check here guards a comparison rather than a crash. Without them a
    string such as proposals="HINTS" was iterated letter by letter and each
    letter built FlatHINTS; a misspelt key was dropped; and HINTS knobs given
    to an experiment without HINTS were ignored -- each one a run whose label
    says it sampled with something it did not.
    """
    cfg = dict(cfg)
    for key, allowed in (("samplers", SAMPLERS), ("proposals", PROPOSALS)):
        value = cfg[key]
        value = [value] if isinstance(value, str) else list(value)
        bad = [v for v in value if v not in allowed]
        if bad or not value:
            raise ValueError(f"{key}={cfg[key]!r}: each entry must be one of {allowed}")
        cfg[key] = value

    unknown = set(cfg) - set(DEFAULT_CFG) - ALL_KNOBS
    if unknown:
        raise ValueError(f"unknown experiment field(s) {sorted(unknown)}; "
                         f"see DEFAULT_CFG and PROPOSAL_KNOBS for the ones that exist")
    # Every knob set must be taken by some proposal the experiment runs.
    taken = {k for p in cfg["proposals"] for k in PROPOSAL_KNOBS.get(p, ())}
    unused = sorted((set(cfg) & ALL_KNOBS) - taken)
    if unused:
        owners = sorted(p for p, knobs in PROPOSAL_KNOBS.items()
                        if set(unused) & set(knobs))
        raise ValueError(f"{unused} only apply to {owners}, which this experiment "
                         f"does not run (proposals={cfg['proposals']})")
    if "hierarchy" in cfg:
        # Checked here rather than left to HINTSProposal, so a bad entry fails
        # on the login node, not in a queued job.
        hierarchy = cfg["hierarchy"]
        if "levels" in cfg or "branching" in cfg:
            raise ValueError("hierarchy replaces levels and branching: set one or the other")
        if not (isinstance(hierarchy, (list, tuple)) and len(hierarchy) >= 2
                and all(isinstance(m, int) for m in hierarchy)
                and hierarchy[0] >= 1 and all(m >= 2 for m in hierarchy[1:])):
            raise ValueError(f"hierarchy={hierarchy!r}: must be [m_0 >= 1, m_1 >= 2, ...]")
        blocks = math.prod(hierarchy)
        if abs(cfg["ss_prop"] * blocks - 1.0) > 1e-9:
            raise ValueError(f"hierarchy {list(hierarchy)} has {blocks} blocks, so "
                             f"ss_prop must be 1/{blocks}, not {cfg['ss_prop']}")
    offset = cfg["chain_offset"]
    if not (isinstance(offset, int) and offset >= 0):
        raise ValueError(f"chain_offset={offset!r}: must be an int >= 0")
    budget, records = cfg["budget"], cfg["records"]
    if budget is not None and not (isinstance(budget, (int, float)) and budget > 0):
        raise ValueError(f"budget={budget!r}: must be a positive number of row "
                         f"reads, or None to run for a number of iterations")
    if not (isinstance(records, int) and records >= 1):
        raise ValueError(f"records={records!r}: must be an int >= 1")
    size = cfg["init_tree_size"]
    if size is not None and (not isinstance(size, int) or size < 0):
        raise ValueError(f"init_tree_size={size!r}: must be an int >= 0, or "
                         f"None to draw the start from the prior")
    if size is not None and cfg["max_tree_size"] is not None and size > cfg["max_tree_size"]:
        raise ValueError(f"init_tree_size={size} is above "
                         f"max_tree_size={cfg['max_tree_size']}")
    rows, name = cfg["init_tree"], cfg["init_tree_name"]
    if rows is None:
        if name is not None:
            raise ValueError(f"init_tree_name={name!r} names an init_tree, "
                             f"and this experiment has none")
    else:
        if size:
            raise ValueError(f"init_tree and init_tree_size={size} both say where "
                             f"to start: set one of the two")
        # Its shape only: whether its features exist and its leaves hold
        # min_samples_leaf rows takes the data, and is checked when a chain
        # builds it.
        try:
            n_nodes = len(idt.tree_rows(rows))
        except ValueError as err:
            raise ValueError(f"init_tree: {err}") from None
        if cfg["max_tree_size"] is not None and n_nodes > cfg["max_tree_size"]:
            raise ValueError(f"init_tree has {n_nodes} internal nodes, above "
                             f"max_tree_size={cfg['max_tree_size']}")
        if name is not None and not (isinstance(name, str)
                                     and INIT_NAME.fullmatch(name)):
            raise ValueError(f"init_tree_name={name!r}: letters, digits and '-' "
                             f"only, as it goes into the result file names")
        if name == PRIOR_INIT:
            raise ValueError(f"init_tree_name={name!r} is what the files of runs "
                             f"started from the prior are tagged with")
    return cfg


def _hms(seconds):
    """Seconds as h:mm:ss."""
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


# How many times run_sweep runs a config before giving up on the experiment.
MAX_ATTEMPTS = 2


def run_sweep(cfg, results_dir, jobs=None):
    """
    Run one experiment's samplers x proposals x chains cross product and write
    <dataset>_<sampler>_<proposal>_<ss_prop * 100>[_init<N>|_init-<name>].h5 into
    results_dir for each (sampler, proposal) pair. This is main()'s body,
    factored out so a driver script can run a list of experiments into one
    results directory -- see run_experiments.py and experiments.py.

    `cfg` is overlaid on DEFAULT_CFG, so any key it does not set keeps its
    default. Two configs that differ only in something the filename does not
    carry write the same files, and the second overwrites the first.
    """
    cfg = validate_cfg(dict(DEFAULT_CFG, **cfg))
    jobs = cfg['jobs'] if jobs is None else jobs

    base = {k: v for k, v in cfg.items() if k not in _SWEEP_KEYS}
    configs = [dict(base, sampler=s, proposal=q, seed=seed)
               for s in cfg['samplers'] for q in cfg['proposals']
               for seed in range(cfg['chain_offset'], cfg['chain_offset'] + cfg['chains'])]

    # Results are held by their position in `configs`, not by the order they
    # finish in, so Run_0 is always seed 0 however the pool schedules them.
    total = len(configs)
    results = [None] * total
    started = time.perf_counter()

    def note(done, index):
        """One self-contained line per finished config, so a long run says
        where it has got to instead of going silent for hours."""
        elapsed = time.perf_counter() - started
        eta = elapsed / done * (total - done)
        c = configs[index]
        tag = f"{c['sampler']}-{c['proposal']}"
        print(f"[{done:>4}/{total}] {tag:<11s} seed {c['seed']:<3} "
              f"{results[index]['cumulative_time'][-1]:7.1f}s sampling   "
              f"elapsed {_hms(elapsed)}  eta {_hms(eta)}", flush=True)

    # A config that raises is run once more before the sweep gives up on it.
    # A seed fixes the whole run, so a retry that succeeds is the run the
    # first attempt would have been; what it guards against is a fault in the
    # worker rather than in the sampler, which would otherwise throw away
    # every other config of a long experiment along with it.
    attempts = [0] * total

    def failed(index, err):
        attempts[index] += 1
        c = configs[index]
        tag = f"{c['sampler']}-{c['proposal']} seed {c['seed']}"
        if attempts[index] >= MAX_ATTEMPTS:
            raise RuntimeError(f"{tag} failed {attempts[index]} times; "
                               f"the last failure is above") from err
        print(f"!!! {tag} failed ({type(err).__name__}: {err}); "
              f"retrying", flush=True)

    if jobs > 1 and total > 1:
        # Retries go to a fresh pool once the current one has drained, because
        # a worker that died outright breaks its pool for every config still
        # in it, and those all have to be run again too.
        pending, done = list(range(total)), 0
        while pending:
            retry = []
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run, configs[i]): i for i in pending}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as err:
                        failed(index, err)
                        retry.append(index)
                        continue
                    done += 1
                    note(done, index)
            pending = retry
    else:
        for index, c in enumerate(configs):
            while results[index] is None:
                try:
                    results[index] = run(c)
                except Exception as err:
                    failed(index, err)
            note(index + 1, index)

    grouped = defaultdict(list)
    for c, res in zip(configs, results):
        grouped[(c['sampler'], c['proposal'])].append((c, res))

    written = []
    for (sampler, proposal), pairs in grouped.items():
        cfgs, runs = zip(*pairs)
        path = os.path.join(results_dir, experiment_filename(
            cfg['dataset'], sampler, proposal, cfg['ss_prop'], init_of(cfg)))
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
    p.add_argument("--proposals", nargs="+", choices=list(PROPOSALS),
                   default=DEFAULT_CFG['proposals'])
    p.add_argument("--chains", type=int, default=DEFAULT_CFG['chains'],
                   help="independent MCMC chains / SMC runs")
    p.add_argument("--iters", type=int, default=DEFAULT_CFG['iters'], help="MCMC iterations")
    p.add_argument("--steps", type=int, default=DEFAULT_CFG['steps'], help="SMC steps")
    p.add_argument("--particles", type=int, default=DEFAULT_CFG['particles'])
    p.add_argument("--store-every", type=int, default=DEFAULT_CFG['store_every'],
                   help="store the sampler's state every k iterations/steps "
                        "(1, every one, is what makes the per-iteration curves "
                        "dense; only distinct states cost anything)")
    p.add_argument("--budget", type=float, default=DEFAULT_CFG['budget'],
                   help="row reads per chain / SMC run: stop each one there "
                        "(--iters and --steps become upper limits) and store "
                        "the states at --records evenly spaced row-read marks")
    p.add_argument("--records", type=int, default=DEFAULT_CFG['records'],
                   help="states stored per run when --budget is set")
    p.add_argument("--no-states", action="store_true",
                   help="skip state recording entirely -- the run can then be "
                        "timed but not evaluated")
    p.add_argument("--no-moves", action="store_true", help="skip the per-move log")
    p.add_argument("--ss-prop", type=float, default=DEFAULT_CFG['ss_prop'])
    p.add_argument("--min-data", type=int, default=DEFAULT_CFG['min_data'])
    p.add_argument("--lam", type=float, default=DEFAULT_CFG['lam'])
    p.add_argument("--min-samples-leaf", type=int, default=DEFAULT_CFG['min_samples_leaf'])
    p.add_argument("--max-tree-size", type=int, default=DEFAULT_CFG['max_tree_size'])
    p.add_argument("--init-tree-size", type=int, default=DEFAULT_CFG['init_tree_size'],
                   help="internal nodes in one starting tree every chain and "
                        "particle shares (0: a stump). Left out, each chain and "
                        "particle draws its own from the prior")
    p.add_argument("--init-tree", default=None, metavar="FILE",
                   help="start from the tree in this JSON file instead: a list "
                        "of [node, left, right, feat, thr] rows, root node 0")
    p.add_argument("--init-tree-name", default=None,
                   help="tag for --init-tree's result files (default: a digest "
                        "of its rows)")
    p.add_argument("--jobs", type=int, default=DEFAULT_CFG['jobs'])
    p.add_argument("--results-root", default="Results")
    p.add_argument("--run-id", default=None,
                   help="results directory name under --results-root "
                        "(default: a timestamp)")
    args = p.parse_args()
    init_tree = None
    if args.init_tree is not None:
        with open(args.init_tree) as f:
            init_tree = json.load(f)

    cfg = dict(DEFAULT_CFG,
               dataset=args.dataset, samplers=args.samplers, proposals=args.proposals,
               chains=args.chains, iters=args.iters, steps=args.steps,
               particles=args.particles, store_every=args.store_every,
               budget=args.budget, records=args.records,
               record_states=not args.no_states,
               record_moves=not args.no_moves, ss_prop=args.ss_prop,
               min_data=args.min_data, lam=args.lam,
               min_samples_leaf=args.min_samples_leaf, max_tree_size=args.max_tree_size,
               init_tree_size=args.init_tree_size, init_tree=init_tree,
               init_tree_name=args.init_tree_name, jobs=args.jobs)

    run_id = resolve_run_id(args.run_id)
    results_dir = os.path.join(args.results_root, run_id)
    os.makedirs(results_dir, exist_ok=True)
    write_run_config(results_dir, dict(cfg, results_root=args.results_root, run_id=run_id))

    run_sweep(cfg, results_dir)

    print(f"\nResults in: {results_dir}"
          f"\n  python examples/incremental_decision_tree/evaluate_results.py --run-id {run_id}"
          f"\n  python examples/incremental_decision_tree/plot_diagnostics.py --run-id {run_id} --name {experiment_name(*experiment_key(cfg))}")


if __name__ == "__main__":
    main()
