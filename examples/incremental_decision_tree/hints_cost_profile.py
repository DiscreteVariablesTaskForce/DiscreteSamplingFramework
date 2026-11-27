"""
Where a HINTS sweep's row reads go.

Starts from trees that budget-run chains ended at (so at the size cap, late in a
run, where most of a budget is spent), runs a few hundred iterations of a
proposal from each, and attributes every row read to the part of the sweep that
made it. A read is charged to the innermost wrapped call it happens in, so the
categories add up to the total:

    labels           drawing the sweep's block labels (NestedBlocks.draw)
    table            the class-count table of every leaf in every unit (tabulate)
    split            splitting a rewritten leaf's rows between children (split)
    rows_on_demand   building a leaf's rows in a block when a move reads them
    screen_propose   proposing a unit move (incl. a grow's split check on full rows)
    screen_eval      the unit vertex's surrogate on its block (eval_move)
    admissibility    a change's partition of the full rows (change_partition)
    unit_apply       applying an accepted unit move to the block state
    unit_other       the rest of the unit step (a grow's routing of full rows)
    handoff          bringing a child forward by earlier siblings' moves
    level_replay     an intermediate vertex replaying its children's moves
    final_replay     the global vertex replaying the composite on every row
    outer            anything outside the proposal (the MCMC accept step)

Run from the repository root, e.g.
    python examples/incremental_decision_tree/hints_cost_profile.py --dataset susy \\
        --starts 4 --iters 300 --ss 0.0625 0.03125 --out Results/overnight
"""
import argparse
import collections
import json
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sampler_diagnostics as sd  # noqa: E402
import discretesampling.domain.incremental_decision_tree as idt  # noqa: E402
from discretesampling.base.algorithms import DiscreteVariableMCMC  # noqa: E402
from discretesampling.domain.incremental_decision_tree.proposals import hints as H  # noqa: E402
from discretesampling.domain.incremental_decision_tree import target as T  # noqa: E402
from discretesampling.domain.incremental_decision_tree.moves import subtree_size  # noqa: E402


class Attributor:
    """Charges row reads to the innermost wrapped call they happen in."""

    def __init__(self, reads):
        self.reads = reads
        self.stack = []
        self.excl = collections.Counter()
        self.calls = collections.Counter()

    def wrap(self, owner, attr, name):
        orig = getattr(owner, attr)
        att = self

        def wrapper(*args, **kwargs):
            label = name(att) if callable(name) else name
            start = att.reads.n
            att.stack.append([label, 0])
            try:
                return orig(*args, **kwargs)
            finally:
                frame = att.stack.pop()
                total = att.reads.n - start
                att.excl[label] += total - frame[1]
                att.calls[label] += 1
                if att.stack:
                    att.stack[-1][1] += total

        setattr(owner, attr, wrapper)
        return orig

    def parent(self):
        return self.stack[-1][0] if self.stack else None


def instrument(att):
    """Wrap every place a HINTS sweep reads rows. Returns the originals."""
    saved = []

    def w(owner, attr, name):
        saved.append((owner, attr, att.wrap(owner, attr, name)))

    w(H.NestedBlocks, "draw", "labels")
    w(H.NestedBlocks, "tabulate", "table")
    w(H.NestedBlocks, "split", "split")
    # The cheaper bookkeeping options (bucket_rows, lazy_counts).
    w(H.BlockRows, "_from_buckets", "bucket_sort")
    w(H.BlockRows, "block_counts", "count_on_demand")

    # Rows built on demand, charged by the step that asked for them.
    def rows_name(a):
        # A block's rows are built from its parent block's, recursively: the
        # whole chain is charged to the step outside it that asked.
        outer = next((f[0] for f in reversed(a.stack)
                      if not f[0].startswith("rows_on_demand")), "?")
        return "rows_on_demand@" + outer
    w(H.BlockRows, "rows", rows_name)
    w(H.HINTSProposal, "_sample", "sweep_top")
    w(H.HINTSProposal, "_node", "vertex")
    w(H.HINTSProposal, "_children", "children")
    w(H.HINTSProposal, "_primitive", "unit_other")
    w(H, "evaluate_subtree_move", "screen_propose")
    w(H, "change_partition", "admissibility")
    w(T.IncrementalTreeTarget, "eval_move", "screen_eval")

    def apply_name(a):
        # A move applied inside a unit step, or replayed by a handoff or a
        # level: charged to that step, so the replays are told apart.
        return "unit_apply" if a.parent() == "unit_other" else a.parent()
    w(H, "apply_subtree_proposal", apply_name)

    def replayed_name(a):
        return "final_replay" if a.parent() == "sweep_top" else "level_replay"
    w(H.HINTSProposal, "_replayed", replayed_name)

    def replay_name(a):
        return a.parent() if a.parent() in ("final_replay", "level_replay") else "handoff"
    w(H.HINTSProposal, "_replay", replay_name)
    return saved


def restore(saved):
    for owner, attr, orig in reversed(saved):
        setattr(owner, attr, orig)


def final_trees(path, chains):
    """[node, left, right, feat, thr] rows of the tree each chain ended at."""
    out = []
    with h5py.File(path, "r") as f:
        for c in chains:
            g = f[f"Run_{c}"]
            lengths = np.asarray(g["state_tree_lengths"])
            slot = int(np.asarray(g["state_index"])[-1, 0])
            start = int(lengths[:slot].sum())
            n = int(lengths[slot])
            cols = [np.asarray(g[k][start:start + n]) for k in
                    ("state_tree_id", "state_tree_left", "state_tree_right",
                     "state_tree_feat", "state_tree_thr")]
            out.append([[int(a), int(b), int(c_), int(d), float(e)]
                        for a, b, c_, d, e in zip(*cols)])
    return out


def profile(dataset, proposal_name, ss, starts, iters, seed, hierarchy=None, **knobs):
    X_train, X_test, y_train, y_test = sd.load_data(dataset)
    cfg = dict(sd.DEFAULT_CFG, dataset=dataset, proposal=proposal_name, ss_prop=ss,
               min_samples_leaf=3, max_tree_size=100, lam=15.0, **knobs)
    if hierarchy is not None:
        cfg["hierarchy"] = hierarchy
    problem, target, proposal = sd.build(cfg, X_train, y_train)
    att = Attributor(problem.reads)
    saved = instrument(att) if proposal_name == "HINTS" else []
    sweeps = []
    try:
        for s, rows in enumerate(starts):
            init = idt.IncrementalTreeInitialProposal(problem, tree=rows)
            last = [problem.reads.n]

            def record(i, current, accepted):
                now = problem.reads.n
                sweeps.append(dict(start=s, reads=now - last[0], accepted=bool(accepted),
                                   n_rho=getattr(proposal, "_n_rho", None),
                                   factors=list(getattr(proposal, "_factors", ())),
                                   level_accepts=getattr(proposal, "n_level_accepts", 0),
                                   level_rejects=getattr(proposal, "n_level_rejects", 0),
                                   inner_moves=getattr(proposal, "n_inner_moves", 0)))
                last[0] = now
                return False

            if proposal_name == "HINTS":
                draw = proposal._draw_root

                def draw_and_note(x, rng, _draw=draw):
                    root, ctx = _draw(x, rng)
                    proposal._n_rho = subtree_size(ctx)
                    return root, ctx
                proposal._draw_root = draw_and_note
            t0 = time.perf_counter()
            mcmc = DiscreteVariableMCMC(idt.IncrementalTree, target, init, proposal=proposal)
            # The start's own routing is not part of any sweep.
            mcmc.sample(1, seed=seed + s, verbose=False)
            last[0] = problem.reads.n
            mcmc.sample(iters, seed=seed + s, verbose=False, callback=record)
            sweeps[-1]["seconds"] = time.perf_counter() - t0
            if proposal_name == "HINTS":
                del proposal._draw_root
    finally:
        restore(saved)
    return att, sweeps, X_train.shape[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="susy")
    p.add_argument("--from-run", default="Results/ch3_budget")
    p.add_argument("--start-file", default=None,
                   help="h5 whose chains' final trees are the starts (default: MH)")
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--ss", type=float, nargs="+", default=[1 / 16, 1 / 32])
    p.add_argument("--proposals", nargs="+", default=["HINTS", "MH", "DA"])
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="Results/overnight")
    p.add_argument("--knob", action="append", default=[],
                   help="extra HINTS setting, e.g. --knob lazy_counts=True (repeatable)")
    p.add_argument("--tag", default="", help="suffix for the report's keys and file")
    args = p.parse_args()

    start_file = args.start_file or os.path.join(
        args.from_run, f"{args.dataset}_mcmc_MH_100_init-prior.h5")
    starts = final_trees(start_file, range(args.starts))
    os.makedirs(args.out, exist_ok=True)
    report = {}
    for name in args.proposals:
        for ss in ([None] if name == "MH" else args.ss):
            knobs = {}
            if name == "HINTS":
                knobs = dict(levels=int(round(np.log2(1 / ss))), branching=2)
                for kv in args.knob:
                    k, v = kv.split("=")
                    knobs[k] = {"True": True, "False": False}.get(v, v)
            att, sweeps, n_train = profile(args.dataset, name, ss if ss else 1.0, starts,
                                           args.iters, args.seed, **knobs)
            reads = np.array([s["reads"] for s in sweeps], float)
            key = f"{name}_{'' if ss is None else round(1 / ss)}{args.tag}"
            entry = dict(iters=len(sweeps), reads_per_iter=float(reads.mean()),
                         reads_per_iter_median=float(np.median(reads)),
                         accept_rate=float(np.mean([s["accepted"] for s in sweeps])),
                         n_train=n_train)
            if name == "HINTS":
                total = sum(att.excl.values())
                entry["breakdown"] = {k: v / len(sweeps) for k, v in att.excl.most_common()}
                entry["breakdown_share"] = {k: v / total for k, v in att.excl.most_common()}
                entry["calls_per_iter"] = {k: v / len(sweeps) for k, v in att.calls.items()}
                n_rho = np.array([s["n_rho"] for s in sweeps], float)
                entry["n_rho_mean"] = float(n_rho.mean())
                entry["n_rho_median"] = float(np.median(n_rho))
                entry["reads_per_n_rho"] = float(reads.sum() / n_rho.sum())
                entry["outer_reads"] = float(reads.sum() - total) / len(sweeps)
            report[key] = entry
            print(f"\n== {key}: {entry['iters']} iterations, {entry['reads_per_iter']:,.0f} reads/iter "
                  f"(median {entry['reads_per_iter_median']:,.0f}), accept {entry['accept_rate']:.3f}")
            if name == "HINTS":
                print(f"   n_rho mean {entry['n_rho_mean']:,.0f}, reads per row below rho "
                      f"{entry['reads_per_n_rho']:.2f}")
                for k, v in entry["breakdown"].items():
                    print(f"   {k:16s} {v:14,.0f} reads/iter  {entry['breakdown_share'][k]:6.1%}")
            with open(os.path.join(args.out, f"profile_{args.dataset}{args.tag}.json"), "w") as f:
                json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
