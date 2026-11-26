"""
Run a list of sampler_diagnostics.py experiments in one go.

List what to run in examples/incremental_decision_tree/experiments.py (or point --file at a copy of it),
then:

    python examples/incremental_decision_tree/run_experiments.py --list
    python examples/incremental_decision_tree/run_experiments.py
    python examples/incremental_decision_tree/run_experiments.py --index 2
    python examples/incremental_decision_tree/run_experiments.py --run-id my_sweep
    python examples/incremental_decision_tree/run_experiments.py --run-id my_sweep --skip-existing

--skip-existing resumes a sweep that stopped partway: every experiment whose
files are already in Results/<run-id>/ is skipped. Within an experiment, a
chain that raises is run once more before the experiment is given up on.

Every experiment in the list writes into the same Results/<run-id>/ directory
-- one run-id for the whole sweep, not one per experiment -- so
`evaluate_results.py --run-id <that>` evaluates all of them in one go and
`plot_diagnostics.py --run-id <that> --name <one of them>` reads any of them
back afterwards. Each (sampler, proposal) of an entry is written to
<dataset>_<sampler>_<proposal>_<ss_prop * 100>[_init<N>|_init-<name>].h5, and two experiments that
would write the same file are refused before anything runs, rather than one
silently overwriting the other partway through a long sweep.

This step only samples and stores trees. No predictive metric is computed
until evaluate_results.py is run over what it wrote; see sampler_diagnostics.py
for why the two are separate.

Experiments run one after another; each one's jobs (default: the CPUs this
process may use, minus one, at most 8; --jobs sets it for every experiment)
parallelises its chains across processes. There is no cross-experiment
parallelism here -- run several of these at once with the same --run-id and
different --index values, which is what slurm/submit.sh does as a SLURM job
array. --count prints the size that array needs.
"""
import argparse
import glob
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_io import (experiment_filename, experiment_names, init_of,  # noqa: E402
                        resolve_run_id, write_run_config)
from sampler_diagnostics import DEFAULT_CFG, prepare_data, run_sweep, validate_cfg  # noqa: E402

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.py")


def load_experiments(path):
    """The EXPERIMENTS list out of a Python file, loaded by path rather than
    import so --file can point anywhere, not just at a sibling of this script.
    `path::NAME` reads GROUPS[NAME] from the file instead, so that one file can
    hold the lists of several runs (each its own --run-id, when their file
    names would otherwise clash)."""
    path, _, group = path.partition("::")
    spec = importlib.util.spec_from_file_location("experiments_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if group:
        groups = getattr(module, "GROUPS", {})
        if group not in groups:
            raise SystemExit(f"{path} has no GROUPS[{group!r}]; it has {sorted(groups)}")
        return groups[group]
    try:
        return module.EXPERIMENTS
    except AttributeError:
        raise SystemExit(f"{path} defines no EXPERIMENTS list") from None


def merge_parts(results_dir):
    """
    Join the parts a --split run wrote under results_dir/parts/part*/ into one
    file per experiment in results_dir, the parts in order, so that Run_k is
    the chain of seed k as in an unsplit run. A file is only joined once every
    part has it; the parts it came from are then deleted. Files already in
    results_dir are left alone.
    """
    import h5py
    parts_root = os.path.join(results_dir, "parts")
    part_dirs = sorted(glob.glob(os.path.join(parts_root, "part*")))
    if not part_dirs:
        print(f"{results_dir}: no parts to merge")
        return
    names = sorted({os.path.basename(f) for d in part_dirs
                    for f in glob.glob(os.path.join(d, "*.h5"))})
    for name in names:
        target = os.path.join(results_dir, name)
        sources = [os.path.join(d, name) for d in part_dirs]
        if os.path.exists(target):
            print(f"{name}: already merged")
            continue
        missing = [s for s in sources if not os.path.exists(s)]
        if missing:
            print(f"!!! {name}: {len(missing)} of {len(sources)} parts missing, not merged "
                  f"({', '.join(os.path.basename(os.path.dirname(m)) for m in missing)})")
            continue
        partial = target + ".partial"
        if os.path.exists(partial):
            os.remove(partial)
        k = 0
        with h5py.File(partial, "w") as out:
            for source in sources:
                with h5py.File(source, "r") as f:
                    runs = sorted((n for n in f if n.startswith("Run_")),
                                  key=lambda n: int(n[len("Run_"):]))
                    for run in runs:
                        f.copy(f[run], out, name=f"Run_{k}")
                        k += 1
        os.replace(partial, target)
        for source in sources:
            os.remove(source)
        print(f"{name}: merged {len(sources)} parts, {k} runs")
    for d in part_dirs:
        if not os.listdir(d):
            os.rmdir(d)
    if os.path.isdir(parts_root) and not os.listdir(parts_root):
        os.rmdir(parts_root)


def experiment_files(entry):
    """The .h5 files one entry writes, one per (sampler, proposal)."""
    cfg = validate_cfg(dict(DEFAULT_CFG, **entry))
    return [experiment_filename(cfg["dataset"], s, p, cfg["ss_prop"], init_of(cfg))
            for s in cfg["samplers"] for p in cfg["proposals"]]


def stored_entries(experiments):
    """The experiments as config.json records them: as given, plus the
    init_tree_size each one runs with, so that a reader never takes a missing
    one for a stump (see results_io.LEGACY_INIT_TREE_SIZE)."""
    return [dict(entry, init_tree_size=dict(DEFAULT_CFG, **entry)["init_tree_size"])
            for entry in experiments]


def check_for_collisions(experiments):
    """Every experiment's files, checked for a repeat before any of them runs:
    two experiments writing the same file would overwrite each other, and
    finding that out after a long sweep has already run the first one is a far
    worse time than before it starts."""
    seen = {}
    for i, entry in enumerate(experiments):
        for filename in experiment_files(entry):
            if filename in seen:
                raise SystemExit(
                    f"experiments {seen[filename]} and {i} would both write "
                    f"{filename!r} -- they differ in nothing the filename "
                    f"carries (dataset, sampler, proposal, ss_prop, "
                    f"starting tree)")
            seen[filename] = i


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default=DEFAULT_FILE,
                   help="Python file defining an EXPERIMENTS list of dicts")
    p.add_argument("--list", action="store_true",
                   help="print the experiments (index, name, overrides) and exit")
    p.add_argument("--index", type=int, default=None,
                   help="run only the n-th experiment of the list (config.json "
                        "still records the whole list, so a SLURM array of these "
                        "sharing one --run-id evaluates as one sweep)")
    p.add_argument("--count", action="store_true",
                   help="check the whole list, print how many experiments it has "
                        "and exit: the size of the SLURM array to run it as")
    p.add_argument("--prepare", action="store_true",
                   help="split every dataset the list uses once, into "
                        "<results-root>/<run-id>/data/, and exit; jobs run with "
                        "DSF_DATA_CACHE pointing there read it instead of the "
                        "CSV (slurm/submit.sh does both)")
    p.add_argument("--jobs", type=int, default=None,
                   help="processes each experiment's chains are spread over, in "
                        "place of each entry's own jobs (on SLURM: the CPUs the "
                        "task was given)")
    p.add_argument("--results-root", default="Results")
    p.add_argument("--run-id", default=None,
                   help="results directory shared by every experiment in this "
                        "sweep, under --results-root (default: a timestamp)")
    p.add_argument("--split", type=int, default=1,
                   help="run each experiment's chains as this many parts, one per "
                        "--index, so that each SLURM task needs only chains/split "
                        "cores: --index i runs part i %% split of experiment "
                        "i // split, and --count counts parts. The parts are "
                        "written under <run>/parts/ and joined by --merge-parts")
    p.add_argument("--merge-parts", action="store_true",
                   help="join the parts a --split run wrote into the usual one "
                        "file per experiment, then delete them, and exit "
                        "(slurm/evaluate.sbatch does this before evaluating)")
    p.add_argument("--skip-existing", action="store_true",
                   help="resume the sweep in --run-id: skip every experiment "
                        "whose .h5 files are all already there (one with only "
                        "some of them is run again in full)")
    args = p.parse_args()
    if args.merge_parts:
        if not args.run_id:
            raise SystemExit("--merge-parts needs --run-id")
        merge_parts(os.path.join(args.results_root, args.run_id))
        return
    if args.split < 1:
        raise SystemExit(f"--split {args.split}: must be at least 1")
    if args.skip_existing and not args.run_id:
        raise SystemExit("--skip-existing needs --run-id: without one the sweep "
                         "writes to a new, empty directory and there is nothing "
                         "to skip")

    experiments = load_experiments(args.file)
    if not experiments:
        raise SystemExit(f"{args.file}: EXPERIMENTS is empty")
    if args.jobs is not None and args.jobs < 1:
        raise SystemExit(f"--jobs {args.jobs}: must be at least 1")

    if args.split > 1:
        uneven = [i for i, e in enumerate(experiments)
                  if dict(DEFAULT_CFG, **e)["chains"] % args.split]
        if uneven:
            raise SystemExit(f"--split {args.split}: the chains of experiment(s) "
                             f"{uneven} do not divide into {args.split} equal parts")
        if args.index is None and not (args.list or args.count or args.prepare):
            raise SystemExit("--split needs --index: each part is its own task")

    # What this invocation runs; the whole list is still checked and recorded.
    selected = list(enumerate(experiments))
    part = None
    if args.index is not None:
        n_tasks = len(experiments) * args.split
        if not 0 <= args.index < n_tasks:
            raise SystemExit(f"--index {args.index} is out of range: this file has "
                             f"{n_tasks} tasks (0-{n_tasks - 1}). "
                             f"Run with --list to see them.")
        experiment, part = divmod(args.index, args.split)
        selected = [selected[experiment]]
        if args.split == 1:
            part = None

    if args.list:
        for i, entry in selected:
            try:
                files = ", ".join(experiment_files(entry))
            except ValueError as err:
                files = f"INVALID: {err}"
            print(f"{i}\t{files}\t{entry}")
        return

    # All of them, before any runs: a bad entry found after hours of the ones
    # ahead of it is the expensive way to find it.
    for i, entry in enumerate(experiments):
        try:
            validate_cfg(dict(DEFAULT_CFG, **entry))
        except ValueError as err:
            raise SystemExit(f"experiment {i}: {err}") from None

    check_for_collisions(experiments)

    if args.count:
        print(len(experiments) * args.split)
        return

    if args.prepare and not args.run_id:
        raise SystemExit("--prepare needs --run-id: the split is written into "
                         "that run's directory")
    run_id = resolve_run_id(args.run_id)
    results_dir = os.path.join(args.results_root, run_id)
    if args.prepare:
        cache_dir = os.path.join(results_dir, "data")
        for dataset in sorted({dict(DEFAULT_CFG, **e)["dataset"] for e in experiments}):
            print(f"prepared {dataset}: {prepare_data(dataset, cache_dir)}")
        return
    os.makedirs(results_dir, exist_ok=True)
    # The whole list, whichever part of it runs here: an array of --index runs
    # writes the same file from every task, and the evaluation reads it back.
    write_run_config(results_dir, {
        "experiments_file": os.path.abspath(args.file),
        "experiments": stored_entries(experiments),
    })

    for i, entry in selected:
        filenames = experiment_files(entry)
        files = ", ".join(filenames)
        merged = all(os.path.exists(os.path.join(results_dir, f)) for f in filenames)
        out_dir = results_dir
        if part is not None:
            # This task's share of the chains, at the seeds an unsplit run
            # would have given them, written beside the other parts.
            chains = dict(DEFAULT_CFG, **entry)["chains"] // args.split
            entry = dict(entry, chains=chains, chain_offset=part * chains)
            out_dir = os.path.join(results_dir, "parts", f"part{part:03d}")
            os.makedirs(out_dir, exist_ok=True)
            files += f" (part {part + 1}/{args.split}: seeds {part * chains}-{(part + 1) * chains - 1})"
        if args.skip_existing and (merged or all(os.path.exists(os.path.join(out_dir, f))
                                                 for f in filenames)):
            print(f"\n# experiment {i + 1}/{len(experiments)}: {files} already written, skipped")
            continue
        print(f"\n{'#' * 78}\n# experiment {i + 1}/{len(experiments)}: {files}\n{'#' * 78}")
        run_sweep(entry, out_dir, jobs=args.jobs)

    names = experiment_names({"experiments": stored_entries(experiments)}, DEFAULT_CFG)
    print(f"\n{len(selected)} experiment(s) complete. Results in: {results_dir}")
    print(f"  python examples/incremental_decision_tree/evaluate_results.py --run-id {run_id}")
    for name in names:
        print(f"  python examples/incremental_decision_tree/plot_diagnostics.py --run-id {run_id} --name {name}")


if __name__ == "__main__":
    main()
