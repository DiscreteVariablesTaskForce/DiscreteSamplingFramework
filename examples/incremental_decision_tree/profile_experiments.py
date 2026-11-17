"""
run_experiments.py over experiments.py, with each experiment under its own
cProfile. Same validation, collision check, config.json and Results/<run-id>/
layout as run_experiments.py; additionally writes, per experiment,
profiles/<run-id>/<stem>.prof, .callgrind and a top-40 .txt summary.

Run from the repo root (dataset paths are relative to it).
"""
import cProfile
import io
import os
import pstats
import sys
import time

EXAMPLES = os.path.join(os.getcwd(), "examples", "incremental_decision_tree")
sys.path.insert(0, EXAMPLES)

import run_experiments as rx  # noqa: E402
from results_io import experiment_names, resolve_run_id, write_run_config  # noqa: E402
from sampler_diagnostics import DEFAULT_CFG, run_sweep, validate_cfg  # noqa: E402

FILE = os.path.join(EXAMPLES, "experiments.py")
experiments = rx.load_experiments(FILE)
for i, entry in enumerate(experiments):
    validate_cfg(dict(DEFAULT_CFG, **entry))
rx.check_for_collisions(experiments)

run_id = resolve_run_id(None)
results_dir = os.path.join("Results", run_id)
prof_dir = os.path.join("profiles", run_id)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(prof_dir, exist_ok=True)
write_run_config(results_dir, {"experiments_file": os.path.abspath(FILE),
                               "experiments": rx.stored_entries(experiments)})

try:
    from pyprof2calltree import convert
except ImportError:
    convert = None

summary = []
for i, entry in enumerate(experiments):
    files = rx.experiment_files(entry)
    stem = files[0][:-len(".h5")]
    print(f"\n{'#' * 78}\n# experiment {i + 1}/{len(experiments)}: {', '.join(files)}\n{'#' * 78}",
          flush=True)
    prof = cProfile.Profile()
    started = time.perf_counter()
    prof.enable()
    try:
        run_sweep(entry, results_dir)
    finally:
        prof.disable()
        wall = time.perf_counter() - started
        base = os.path.join(prof_dir, stem)
        prof.dump_stats(base + ".prof")
        stats = pstats.Stats(prof)
        if convert is not None:
            convert(stats, base + ".callgrind")
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
        buf.write("\n\n")
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(40)
        with open(base + ".txt", "w") as fh:
            fh.write(buf.getvalue())
        summary.append((stem, wall))
        print(f"--> profile: {base}.prof  ({wall:.1f}s wall, profiled)", flush=True)

print(f"\nAll {len(experiments)} experiment(s) complete. Results in: {results_dir}")
print(f"Profiles in: {prof_dir}")
for stem, wall in summary:
    print(f"  {wall:8.1f}s  {stem}")
print(f"  python examples/incremental_decision_tree/evaluate_results.py --run-id {run_id}")
for name in experiment_names({"experiments": rx.stored_entries(experiments)}, DEFAULT_CFG):
    print(f"  python examples/incremental_decision_tree/plot_diagnostics.py --run-id {run_id} --name {name}")
