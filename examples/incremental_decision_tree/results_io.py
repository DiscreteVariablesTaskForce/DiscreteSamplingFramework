"""
HDF5 persistence for sampler_diagnostics.py runs.

One results directory per sweep, named explicitly or by timestamp:

    Results/<run-id>/
        config.json                  the CLI args the sweep was run with
        wine_mcmc_MH_100.h5          one file per (dataset, sampler, proposal,
        wine_mcmc_DA_10.h5           ss_prop * 100), '.' written as 'p'
        wine_mcmc_HINTS_12p5.h5
        wine_mcmc_HINTS_12p5_metrics.npz   evaluate_results.py's, beside it
        wine_mcmc_HINTS_12p5_init10.h5     started from a 10-node random tree
        wine_mcmc_HINTS_12p5_init-mine.h5  started from a given tree, named 'mine'
        wine_smc_MH_100.h5
        ...

Inside one .h5, one group per chain (MCMC) or per independent run (SMC):
Run_0, Run_1, .... A run's scalars (acceptance rate, evaluation counters, ...)
are stored as group attributes; its per-iteration / per-move columns (n_nodes,
the move log, ...) and the flat state arrays are stored as compressed datasets.
load_experiment_hdf5 hands both back merged into the same {column: array} dict
sampler_diagnostics.py's run() produces, so nothing downstream
(plot_diagnostics.py, evaluate_results.py) needs to know results came off disk
rather than out of a fresh run.

The `state_*` columns are the trees the run visited -- one entry per distinct
state, plus a per-record index into them. `states_of(run)` turns them back into
a StateSeries; see the domain's states.py for the layout and for why only
distinct states are held.

This is the same shape as the cluster harness's Results/<run_id>/<experiment>.h5
layout, cut down to what a diagnostics run on one machine needs: no chunked
streaming (a batch finishes before anything is written), no run-length
encoding of states, no format versioning. If a run ever grows past what fits
in memory during a batch, that machinery is what to reach for.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime

import h5py
import numpy as np

from discretesampling.domain.incremental_decision_tree.states import series_from_arrays

RUN_PREFIX = "Run_"

# The fixed sweep dimensions a results directory is named over. Everything that
# goes looking for an experiment's files checks this cross product rather than
# globbing: "HINTS" is a suffix of "FlatHINTS", so a pattern loose enough to
# glob with can match a file that belongs to some other proposal.
#
# These are the one list of each: sampler_diagnostics.py takes the proposals it
# will accept from here, and plot_diagnostics.py the pairs it looks for. A
# second copy elsewhere is how a proposal ends up sampled but never evaluated --
# the .h5 files are written, nothing goes looking for them, and the run just
# comes out missing a method with no error anywhere.
SAMPLERS = ("mcmc", "smc")
PROPOSALS = ("MH", "DA", "FlatHINTS", "HINTS")


# The proposals that never subsample. They ignore ss_prop, so their files are
# tagged 100 -- the whole training set -- whatever ss_prop the experiment
# carried, rather than a level they were never run at.
FULL_DATA_PROPOSALS = ("MH",)


def ss_tag(ss_prop):
    """ss_prop as a percentage fit for a filename: 0.25 -> '25', 0.125 -> '12p5'."""
    return f"{round(100 * ss_prop, 6):g}".replace(".", "p")


# What a run started from, as names carry it: the starting tree's size (an
# int; 0 is a stump) for a random tree, or a label (a str) for a given one.
# A label has no '_' in it, which is what the names are split on.
INIT_NAME = re.compile(r"[A-Za-z0-9-]+")


def init_of(cfg):
    """
    What a config's runs start from: init_tree_size, or for a given init_tree
    its init_tree_name -- by default a digest of its rows, so that two
    different trees never share a name and the same tree always has the same
    one, whichever order its rows were written in.
    """
    rows = cfg.get("init_tree")
    if rows is None:
        return cfg.get("init_tree_size", 0)
    if cfg.get("init_tree_name"):
        return cfg["init_tree_name"]
    canonical = sorted([int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])]
                       for r in rows)
    return "t" + hashlib.sha1(json.dumps(canonical).encode()).hexdigest()[:8]


def init_tag(init):
    """The starting tree as a name suffix: '' for a stump, so names from
    before there was a choice still mean what they meant, '_init10' for a
    random 10-node tree, '_init-mine' for a given tree named 'mine'."""
    if isinstance(init, str):
        return f"_init-{init}"
    return f"_init{int(init)}" if init else ""


def experiment_stem(dataset, sampler, proposal, ss_prop, init=0):
    """{dataset}_{sampler}_{proposal}_{ss_prop * 100}[_init{N}|_init-{name}],
    the name both an experiment's .h5 and its metrics .npz are built on."""
    if proposal in FULL_DATA_PROPOSALS:
        ss_prop = 1.0
    return f"{dataset}_{sampler}_{proposal}_{ss_tag(ss_prop)}{init_tag(init)}"


def experiment_filename(dataset, sampler, proposal, ss_prop, init=0):
    return f"{experiment_stem(dataset, sampler, proposal, ss_prop, init)}.h5"


def metrics_filename(dataset, sampler, proposal, ss_prop, init=0):
    return f"{experiment_stem(dataset, sampler, proposal, ss_prop, init)}_metrics.npz"


# An experiment, as evaluate_results.py and plot_diagnostics.py know it, is
# every (sampler, proposal) run on one dataset at one subsample level from one
# starting tree, named {dataset}_{ss_prop * 100}[_init{N}|_init-{name}]. The
# full-data proposals belong to every level -- they are the reference each
# level is compared against -- so an experiment holding only those is at 100.

def experiment_key(cfg):
    """(dataset, ss_prop, init) of a config already overlaid on DEFAULT_CFG;
    see init_of for the last."""
    proposals = cfg["proposals"]
    proposals = [proposals] if isinstance(proposals, str) else proposals
    if all(p in FULL_DATA_PROPOSALS for p in proposals):
        return cfg["dataset"], 1.0, init_of(cfg)
    return cfg["dataset"], cfg["ss_prop"], init_of(cfg)


def experiment_name(dataset, ss_prop, init=0):
    return f"{dataset}_{ss_tag(ss_prop)}{init_tag(init)}"


def parse_experiment(name):
    """experiment_name's inverse, (dataset, ss_prop, init); ValueError if
    `name` is not one experiment_name could have produced."""
    rest, _, tag = name.rpartition("_")
    init = 0
    try:
        if tag.startswith("init"):
            init = tag[len("init"):]
            if init.startswith("-") and INIT_NAME.fullmatch(init[1:]):
                init = init[1:]
            else:
                init = int(init)
            rest, _, tag = rest.rpartition("_")
        if not rest:
            raise ValueError
        return rest, float(tag.replace("p", ".")) / 100, init
    except ValueError:
        raise ValueError(f"{name!r} is not a <dataset>_<ss_prop * 100>"
                         f"[_init<N>|_init-<name>] experiment name, e.g. "
                         f"covtype_12p5, covtype_12p5_init10 or "
                         f"covtype_12p5_init-mine") from None


def parse_experiment_name(name):
    """(dataset, ss_prop) of an experiment name, its starting tree dropped;
    parse_experiment keeps it."""
    return parse_experiment(name)[:2]


def find_experiment_files(results_dir, name):
    """{(sampler, proposal): path} for the experiment called `name`, the
    full-data proposals' files included. {} if `name` is malformed."""
    try:
        dataset, ss_prop, init = parse_experiment(name)
    except ValueError:
        return {}
    out = {}
    for sampler in SAMPLERS:
        for proposal in PROPOSALS:
            path = os.path.join(results_dir, experiment_filename(
                dataset, sampler, proposal, ss_prop, init))
            if os.path.exists(path):
                out[(sampler, proposal)] = path
    return out


def experiment_names(config, defaults):
    """
    The experiment names a results directory's config.json covers -- what an
    evaluation or a plot runs over when nothing is named, and what a "no such
    experiment" message needs to list.

    run_experiments.py writes the whole 'experiments' list it was given; a bare
    sampler_diagnostics.py sweep writes the single config it ran. Each is
    overlaid on `defaults` (DEFAULT_CFG) for whatever it left out. [] for a
    directory that predates config.json, or one hand-populated with .h5 files.
    """
    entries = config.get("experiments") or ([config] if "dataset" in config else [])
    return sorted({experiment_name(*experiment_key(dict(defaults, **entry)))
                   for entry in entries})


def resolve_run_id(run_id=None):
    """Results directory name: given explicitly, or a timestamp."""
    return run_id if run_id else datetime.now().strftime("%y%m%d_%H%M%S")


def latest_run_id(results_root):
    """The most recently written run directory under `results_root`, for a
    caller that wants 'whatever I ran last' without naming it. None if
    `results_root` holds no run directories (or does not exist)."""
    if not os.path.isdir(results_root):
        return None
    dirs = [d for d in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, d))]
    if not dirs:
        return None
    return max(dirs, key=lambda d: os.path.getmtime(os.path.join(results_root, d)))


def _is_scalar(value):
    return np.ndim(value) == 0


def save_run_hdf5(path, run_idx, result):
    """Append one run's {column: value} dict as a group in `path`, creating the
    file if it does not exist yet. Scalars (acceptance_rate, the target's
    evaluation counters, ...) become attributes; arrays become datasets,
    compressed unless empty (compression needs at least one chunk)."""
    with h5py.File(path, 'a') as f:
        grp = f.require_group(f"{RUN_PREFIX}{run_idx}")
        for key, value in result.items():
            if _is_scalar(value):
                grp.attrs[key] = value
                continue
            arr = np.asarray(value)
            if arr.size == 0:
                grp.create_dataset(key, data=arr)
            else:
                grp.create_dataset(key, data=arr, compression="lzf", shuffle=True)


def save_experiment_hdf5(path, runs):
    """Write every run of one (dataset, sampler, proposal) experiment to
    `path`, replacing it if it already exists.

    The runs are written to a temporary file that is renamed onto `path` only
    once all of them are in, so `path` is never a partial write: a sweep
    resumed with run_experiments.py --skip-existing takes any file that exists
    as finished."""
    partial = path + ".partial"
    if os.path.exists(partial):
        os.remove(partial)
    for i, result in enumerate(runs):
        save_run_hdf5(partial, i, result)
    os.replace(partial, path)


def load_experiment_hdf5(path):
    """The runs written by save_experiment_hdf5, as a list of {column: array}
    dicts in Run_0, Run_1, ... order -- the same shape run() itself returns."""
    out = []
    with h5py.File(path, 'r') as f:
        names = sorted((k for k in f if k.startswith(RUN_PREFIX)),
                       key=lambda k: int(k[len(RUN_PREFIX):]))
        for name in names:
            grp = f[name]
            run = {key: grp.attrs[key] for key in grp.attrs}
            run.update({key: grp[key][()] for key in grp.keys()})
            out.append(run)
    return out


def states_of(run):
    """
    The states one run visited, as a StateSeries -- None if it was sampled
    with --no-states and so has none.

    `run` is one entry of load_experiment_hdf5's list, or one of the dicts
    sampler_diagnostics.run() returns directly; the state arrays are the same
    either way, so an evaluation can be run against a fresh result without a
    round trip through disk.
    """
    return series_from_arrays(run)


def stored_tree(path, run=0, record=0, particle=None):
    """
    One tree out of a results file, as the [node, left, right, feat, thr] rows
    an experiment takes as its init_tree: the tree `particle` held at stored
    record `record` of Run_<run>. Left unnamed, the particle is the one
    carrying the most weight there (the only one, for MCMC).

    Only that run's states are read, not the whole file.
    """
    with h5py.File(path, 'r') as f:
        grp = f[f"{RUN_PREFIX}{run}"]
        arrays = {key: grp.attrs[key] for key in grp.attrs}
        arrays.update({key: grp[key][()] for key in grp.keys()
                       if key.startswith("state_")})
    series = series_from_arrays(arrays)
    if series is None:
        raise ValueError(f"{path}: Run_{run} was stored without states")
    if particle is None:
        weights = series.weights(record)
        particle = 0 if weights is None else int(np.argmax(weights))
    tree = series.state(series.slots(record)[particle]).tree
    return [[int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])] for r in tree]


def write_run_config(results_dir, config):
    """config.json alongside the .h5 files: what the sweep that wrote this
    directory was asked to run, for reference when loading it back later."""
    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2, default=str)


def read_run_config(results_dir):
    """The config.json beside a results directory's .h5 files, or {} if the
    directory predates it (or was written by something else)."""
    path = os.path.join(results_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
