"""cProfile one run of each sampler x proposal, in-process, and save a .prof per pair."""
import argparse
import cProfile
import functools
import os
import pstats
import sys
from pyprof2calltree import convert

sys.path.insert(0, "examples/incremental_decision_tree")
import sampler_diagnostics as sd  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="covtype")
p.add_argument("--samplers", nargs="+", default=["mcmc", "smc"])
p.add_argument("--proposals", nargs="+", default=["MH", "DA", "FlatHINTS", "HINTS"])
p.add_argument("--iters", type=int, default=1000)
p.add_argument("--steps", type=int, default=100)
p.add_argument("--particles", type=int, default=50)
p.add_argument("--ss-prop", type=float, default=0.1)
p.add_argument("--max-tree-size", type=int, default=100)
p.add_argument("--record", action="store_true",
               help="keep state and move recording on, to profile the diagnostics too")
p.add_argument("--out", default="profiles")
p.add_argument("--top", type=int, default=20)
args = p.parse_args()

cfg = dict(sd.DEFAULT_CFG, dataset=args.dataset, iters=args.iters, steps=args.steps,
           particles=args.particles, ss_prop=args.ss_prop, max_tree_size=args.max_tree_size,
           record_states=args.record, record_moves=args.record, seed=0)

# Load and split the data once, before any profiling starts.
sd.load_data = functools.cache(sd.load_data)
sd.load_data(args.dataset)
os.makedirs(args.out, exist_ok=True)

for sampler in args.samplers:
    for proposal in args.proposals:
        profiler = cProfile.Profile()
        profiler.runcall(sd.run, dict(cfg, sampler=sampler, proposal=proposal))
        path = os.path.join(args.out, f"{args.dataset}_{sampler}_{proposal}.prof")
        profiler.dump_stats(path)
        # also save a calltree for kcachegrind
        convert(profiler.getstats(), path.replace(".prof", ".callgrind"))
        print(f"\n{'=' * 30} {sampler.upper()} {proposal}  ->  {path}")
        pstats.Stats(profiler).sort_stats("tottime").print_stats(args.top)

