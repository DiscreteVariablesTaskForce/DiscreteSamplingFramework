"""
Choose the tree-size cap for the large-tree runs from the pilot's chains.

Reads every .h5 of the pilot run, takes the largest tree (decision nodes) any
chain reached at any record, and sets

    cap = max(FLOOR, SAFETY x largest), rounded up to a round number,

so that no chain of the main runs, which use the same settings and budgets, should
reach it. If the pilot itself hit its safety cap, the trees were still growing:
the cap is then the pilot's cap, and the summary says that it binds. Writes the
decision (and the evidence for it) to <pilot dir>/tree_cap.json, which
experiments_bigtree.py reads, and exits non-zero if the pilot has no chains, so
that the main runs held on this job never start.

    python examples/incremental_decision_tree/decide_tree_cap.py --pilot-dir Results/bigtree_pilot
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import h5py
import numpy as np

STORE_BYTES_PER_NODE = {"susy": 40, "hepmass": 40, "digits": 73, "sensorless": 77}


def nice_ceil(x):
    """The smallest of 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 (x 10^k) not below x."""
    k = 10 ** math.floor(math.log10(x))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * k >= x:
            return int(round(m * k))
    return int(round(10 * k))


def chains_of(path):
    """(chain, sizes, row_reads, wall seconds) for every chain of one file."""
    with h5py.File(path) as f:
        for key in sorted(f.keys(), key=lambda s: int(s.split("_")[1])):
            run = f[key]
            yield (key, np.asarray(run["n_nodes"][()]), np.asarray(run["row_reads"][()]),
                   float(run["cumulative_time"][()][-1]) if "cumulative_time" in run else float("nan"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot-dir", default="Results/bigtree_pilot")
    p.add_argument("--out", default=None, help="default: <pilot dir>/tree_cap.json")
    p.add_argument("--safety", type=float, default=1.5)
    p.add_argument("--floor", type=int, default=1000)
    p.add_argument("--pilot-cap", type=int, default=None,
                   help="the pilot's own cap (default: read from its config.json)")
    args = p.parse_args()
    out_path = args.out or os.path.join(args.pilot_dir, "tree_cap.json")

    pilot_cap = args.pilot_cap
    config = os.path.join(args.pilot_dir, "config.json")
    if pilot_cap is None and os.path.exists(config):
        with open(config) as f:
            stored = json.load(f)
        entries = stored["experiments"] if isinstance(stored, dict) and "experiments" in stored else stored
        caps = {e.get("max_tree_size") for e in entries}
        pilot_cap = max(c for c in caps if c is not None) if caps - {None} else None

    files = sorted(glob.glob(os.path.join(args.pilot_dir, "*.h5")))
    rows, largest = [], 0
    for path in files:
        name = os.path.basename(path)
        for chain, sizes, reads, wall in chains_of(path):
            if len(sizes) == 0:
                continue
            top = int(sizes.max())
            largest = max(largest, top)
            # How far into the run the chain got within 10% of its largest tree:
            # near 1 means it was still growing at the end of the budget.
            at = reads[np.argmax(sizes >= 0.9 * top)] / reads[-1] if reads[-1] > 0 else float("nan")
            rows.append((name, chain, int(sizes[-1]), top, at, wall / 3600))
    if not rows:
        sys.exit(f"no pilot chains found in {args.pilot_dir}: not writing a cap")

    binding = pilot_cap is not None and largest >= pilot_cap
    cap = pilot_cap if binding else nice_ceil(max(args.floor, args.safety * largest))

    print(f"{'file':42s} {'chain':>7s} {'final':>7s} {'largest':>8s} {'90% of it at':>13s} {'hours':>6s}")
    for name, chain, final, top, at, hours in rows:
        print(f"{name:42s} {chain:>7s} {final:7d} {top:8d} {at:12.0%} {hours:6.1f}")
    print(f"\nlargest tree in the pilot: {largest} nodes (pilot cap {pilot_cap})")
    print(f"tree cap for the main runs: {cap}" + ("  -- the pilot hit its own cap: trees were still growing,"
                                                  " so this cap will bind" if binding else ""))
    still_growing = sum(at > 0.9 for *_, at, _ in rows)
    if still_growing:
        print(f"note: {still_growing} of {len(rows)} chains reached 90% of their largest tree only in the last"
              " 10% of the budget, i.e. they were still growing")
    # Worst-case storage of the stored trees at this cap (every record a new tree).
    est = {d: cap * b for d, b in STORE_BYTES_PER_NODE.items()}
    print("worst-case stored trees per chain (MCMC, 500 records): "
          + ", ".join(f"{d} {500 * v / 1e6:.0f} MB" for d, v in est.items()))

    decision = dict(cap=int(cap), largest=int(largest), safety=args.safety, floor=args.floor,
                    pilot_cap=pilot_cap, binding=bool(binding), chains=len(rows), files=len(files),
                    still_growing=int(still_growing), decided=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
