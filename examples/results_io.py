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
Run_0, Run_1, .... A run's scalars (acceptance rate, evaluation counters, a
final confusion matrix's mean, ...) are stored as group attributes; its
per-iteration / per-move columns (n_nodes, the move log, ...) are stored as
compressed datasets. load_experiment_hdf5 hands both back merged into the same
{column: array} dict sampler_diagnostics.py's run() produces, so nothing
downstream (plot_diagnostics.py included) needs to know results came off disk
rather than out of a fresh run.

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

RUN_PREFIX = "Run_"


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
