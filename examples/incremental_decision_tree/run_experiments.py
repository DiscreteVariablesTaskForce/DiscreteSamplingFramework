"""
Run a list of sampler_diagnostics.py experiments in one go.

List what to run in examples/incremental_decision_tree/experiments.py (or point --file at a copy of it),
then:

    python examples/incremental_decision_tree/run_experiments.py --list
    python examples/incremental_decision_tree/run_experiments.py
    python examples/incremental_decision_tree/run_experiments.py --index 2
    python examples/incremental_decision_tree/run_experiments.py --run-id my_sweep

Every experiment in the list writes into the same Results/<run-id>/ directory
-- one run-id for the whole sweep, not one per experiment -- so
`evaluate_results.py --run-id <that>` evaluates all of them in one go and
`plot_diagnostics.py --run-id <that> --name <one of them>` reads any of them
back afterwards. Two experiments that would write the same .h5 files (the same
dataset and no distinct 'name') are refused before anything runs, rather than
one silently overwriting the other partway through a long sweep.

This step only samples and stores trees. No predictive metric is computed
until evaluate_results.py is run over what it wrote; see sampler_diagnostics.py
for why the two are separate.

Experiments run one after another; each one's own --jobs (default: this
machine's cores minus a couple, same as sampler_diagnostics.py) parallelises
its chains across processes. There is no cross-experiment parallelism here --
run two of these at once, with the same --run-id, if that is wanted.
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_io import resolve_run_id, write_run_config  # noqa: E402
from sampler_diagnostics import DEFAULT_CFG, run_sweep  # noqa: E402

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.py")


def load_experiments(path):
    """The EXPERIMENTS list out of a Python file, loaded by path rather than
    import so --file can point anywhere, not just at a sibling of this script."""
    spec = importlib.util.spec_from_file_location("experiments_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.EXPERIMENTS
    except AttributeError:
        raise SystemExit(f"{path} defines no EXPERIMENTS list") from None


def experiment_name(entry):
    return entry.get("name") or entry.get("dataset", DEFAULT_CFG["dataset"])


def check_for_collisions(experiments):
    """Every experiment's name, checked for a repeat before any of them runs:
    two experiments sharing a name would overwrite each other's .h5 files, and
    finding that out after a long sweep has already run the first one is a far
    worse time than before it starts."""
    seen = {}
    for i, entry in enumerate(experiments):
        name = experiment_name(entry)
        if name in seen:
            raise SystemExit(
                f"experiments {seen[name]} and {i} would both write "
                f"'{name}_*.h5' -- give one of them a distinct 'name'")
        seen[name] = i


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=DEFAULT_FILE,
                   help="Python file defining an EXPERIMENTS list of dicts")
    p.add_argument("--list", action="store_true",
                   help="print the experiments (index, name, overrides) and exit")
    p.add_argument("--index", type=int, default=None,
                   help="run only the n-th experiment of the list")
    p.add_argument("--results-root", default="Results")
    p.add_argument("--run-id", default=None,
                   help="results directory shared by every experiment in this "
                        "sweep, under --results-root (default: a timestamp)")
    args = p.parse_args()

    experiments = load_experiments(args.file)
    if not experiments:
        raise SystemExit(f"{args.file}: EXPERIMENTS is empty")

    if args.list:
        for i, entry in enumerate(experiments):
            overrides = {k: v for k, v in entry.items() if k != "name"}
            print(f"{i}\t{experiment_name(entry)}\t{overrides}")
        return

    if args.index is not None:
        if not 0 <= args.index < len(experiments):
            raise SystemExit(f"--index {args.index} is out of range: this file has "
                             f"{len(experiments)} experiments (0-{len(experiments) - 1}). "
                             f"Run with --list to see them.")
        experiments = [experiments[args.index]]

    check_for_collisions(experiments)

    run_id = resolve_run_id(args.run_id)
    results_dir = os.path.join(args.results_root, run_id)
    os.makedirs(results_dir, exist_ok=True)
    write_run_config(results_dir, {
        "experiments_file": os.path.abspath(args.file),
        "experiments": experiments,
    })

    for i, entry in enumerate(experiments):
        name = experiment_name(entry)
        print(f"\n{'#' * 78}\n# experiment {i + 1}/{len(experiments)}: {name}\n{'#' * 78}")
        run_sweep(entry, results_dir, name=name)

    names = sorted({experiment_name(e) for e in experiments})
    print(f"\nAll {len(experiments)} experiment(s) complete. Results in: {results_dir}")
    print(f"  python examples/incremental_decision_tree/evaluate_results.py --run-id {run_id}")
    for name in names:
        print(f"  python examples/incremental_decision_tree/plot_diagnostics.py --run-id {run_id} --name {name}")


if __name__ == "__main__":
    main()
