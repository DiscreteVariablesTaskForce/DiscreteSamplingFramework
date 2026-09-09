"""
HDF5 persistence for sampler_diagnostics.py runs.

One results directory per sweep, named explicitly or by timestamp:

    Results/<run-id>/
        config.json                  the CLI args the sweep was run with
        wine_mcmc_MH.h5               one file per (dataset, sampler, proposal)
        wine_mcmc_DA.h5
        wine_mcmc_HINTS.h5
        wine_smc_MH.h5
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

import json
import os
from datetime import datetime

import h5py
import numpy as np

from discretesampling.domain.incremental_decision_tree.states import series_from_arrays

RUN_PREFIX = "Run_"

# The fixed sweep dimensions a results directory is named over. Everything that
# goes looking for an experiment's files checks this cross product rather than
# globbing: two experiment names can share a prefix (a dataset "wine" and a
# second experiment named "wine_deep", say), and "wine_deep_mcmc_HINTS.h5" is
# indistinguishable from a "wine" experiment using a sampler called "deep_mcmc"
# from the filename's shape alone.
SAMPLERS = ("mcmc", "smc")
PROPOSALS = ("MH", "DA", "HINTS")


def experiment_filename(name, sampler, proposal):
    return f"{name}_{sampler}_{proposal}.h5"


def find_experiment_files(results_dir, name):
    """{(sampler, proposal): path} for the experiment called `name`."""
    out = {}
    for sampler in SAMPLERS:
        for proposal in PROPOSALS:
            path = os.path.join(results_dir,
                                experiment_filename(name, sampler, proposal))
            if os.path.exists(path):
                out[(sampler, proposal)] = path
    return out


def experiment_datasets(config, default_dataset):
    """
    {experiment name: the dataset it was run on}, from a results directory's
    config.json -- which is what an evaluation needs to know to reload the
    right rows, and what a "no such experiment" message needs to list.

    run_experiments.py writes the whole 'experiments' list it was given; a bare
    sampler_diagnostics.py sweep writes the single config it ran, whose name is
    its dataset. {} for a directory that predates config.json, or one
    hand-populated with .h5 files.
    """
    if "experiments" in config:
        out = {}
        for entry in config["experiments"]:
            dataset = entry.get("dataset", default_dataset)
            out[entry.get("name") or dataset] = dataset
        return out
    dataset = config.get("dataset")
    return {dataset: dataset} if dataset else {}


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
    `path`, replacing it if it already exists."""
    if os.path.exists(path):
        os.remove(path)
    for i, result in enumerate(runs):
        save_run_hdf5(path, i, result)


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
