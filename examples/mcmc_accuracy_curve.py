"""
Per-iteration test accuracy, averaged over many independent MCMC chains.

Each chain is run with its own seed; at every iteration the current tree's test
accuracy is recorded, and the curves are averaged across chains iteration by
iteration. That shows how fast a chain climbs and where it plateaus -- which a
single chain, or an accuracy read off the final samples, does not.

    python examples/mcmc_accuracy_curve.py --chains 100 --iters 20000

Making long chains affordable:

  * the accuracy is only recomputed when a move is accepted. A rejected step
    leaves the state untouched, so its accuracy is the previous value. Pass
    --no-cache to recompute every iteration and check that this holds.

Chains are independent, so they run in parallel across processes.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from discretesampling.base.algorithms import DiscreteVariableMCMC
from discretesampling.domain import decision_tree as dt
from discretesampling.domain import incremental_decision_tree as idt


def load_data():
    X, y = datasets.fetch_covtype(return_X_y=True)
    y = pd.Categorical(y).codes
    return train_test_split(X, y, train_size=0.8, stratify=y, random_state=0)


def build(domain, X_train, y_train, lam, min_samples_leaf, max_tree_size, proposal, ss_prop, min_data):
    """Returns (mcmc, accuracy_fn). accuracy_fn is a fraction in [0, 1] in both
    domains -- dt.accuracy returns a percentage, so it is rescaled here."""
    if domain == "incremental":
        problem = idt.IncrementalTreeProblem(
            X_train, y_train, lam=lam, min_samples_leaf=min_samples_leaf,
            max_tree_size=max_tree_size)
        target = idt.IncrementalTreeTarget(problem)
        if proposal == "MH":
            proposal = idt.IncrementalTreeProposal()
        elif proposal == "DA":
            proposal = idt.DAProposal(target=target, ss_prop=ss_prop, min_data=min_data)
        elif proposal == "HINTS":
            proposal = idt.HINTSProposal(target=target, ss_prop=ss_prop, min_data=min_data)
        mcmc = DiscreteVariableMCMC(
            idt.IncrementalTree, target,
            idt.IncrementalTreeInitialProposal(problem),
            proposal=proposal)

        def accuracy(state, X_test, y_test):
            return idt.accuracy(y_test, idt.predict([state], X_test))
    else:
        mcmc = DiscreteVariableMCMC(dt.Tree, dt.TreeTarget(lam, None),
                                    dt.TreeInitialProposal(X_train, y_train))

        def accuracy(state, X_test, y_test):
            labels = dt.stats([state], X_test).predict(X_test, use_majority=True)
            return dt.accuracy(y_test, labels) / 100.0

    return mcmc, accuracy


def one_chain(job):
    """Run a single chain, returning its per-iteration accuracy curve."""
    domain, seed, iters, lam, msl, cap, proposal, ss_prop, min_data, use_cache = job
    X_train, X_test, y_train, y_test = load_data()
    mcmc, accuracy = build(domain, X_train, y_train, lam, msl, cap, proposal, ss_prop, min_data)

    curve = np.empty(iters, dtype=np.float32)
    last = np.float32(np.nan)
    last_state = None

    def record(i, current, accepted):
        nonlocal last, last_state
        # Recompute only when the chain is on a different state object. That
        # covers rejections, and also the incremental domain's "stay" moves,
        # which are *accepted* but hand back the very same object -- keying on
        # `accepted` alone misses those and buys nothing there.
        if current is not last_state or not use_cache:
            last = np.float32(accuracy(current, X_test, y_test))
            last_state = current
        curve[i] = last

    mcmc.sample(iters, seed=seed, verbose=False)
    return curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["incremental", "classic"],
                   default="incremental")
    p.add_argument("--proposal", choices=["MH", "DA", "HINTS"], default="HINTS")
    p.add_argument("--ss-prop", type=float, default=0.1)
    p.add_argument("--min-data", type=int, default=200)
    p.add_argument("--chains", type=int, default=10)
    p.add_argument("--iters", type=int, default=10_000)
    p.add_argument("--lam", type=float, default=15.0)
    p.add_argument("--min-samples-leaf", type=int, default=3)
    p.add_argument("--max-tree-size", type=int, default=10)
    p.add_argument("--jobs", type=int, default=min(14, os.cpu_count()-2 or 1))
    p.add_argument("--no-cache", action="store_true",
                   help="recompute accuracy every iteration (slow; correctness check)")
    p.add_argument("--out", default="accuracy_curves.npy")
    p.add_argument("--plot", default="accuracy_curve_HINTS.png")
    args = p.parse_args()

    jobs = [(args.domain, seed, args.iters, args.lam, args.min_samples_leaf,
             args.max_tree_size, args.proposal, args.ss_prop, args.min_data, not args.no_cache)
            for seed in range(args.chains)]

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            curves = list(pool.map(one_chain, jobs, chunksize=1))
    else:
        curves = [one_chain(j) for j in jobs]

    curves = np.vstack(curves)                 # (chains, iters)
    mean = curves.mean(axis=0)                 # the average asked for
    np.save(args.out, curves)

    print(f"{args.domain}: {curves.shape[0]} chains x {curves.shape[1]} iterations")
    print(f"saved per-chain curves to {args.out}  {curves.shape} float32")
    print("\n%10s %10s %10s %10s" % ("iter", "mean acc", "sd across", "se"))
    marks = [m for m in (0, 10, 100, 500, 1000, 5000, 10_000, 15_000,
                         args.iters - 1) if m < args.iters]
    for m in marks:
        col = curves[:, m]
        print("%10d %10.4f %10.4f %10.4f"
              % (m, col.mean(), col.std(), col.std() / np.sqrt(len(col))))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lo, hi = np.percentile(curves, [25, 75], axis=0)
        it = np.arange(curves.shape[1])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(it, lo, hi, alpha=0.25,
                        label="interquartile range across chains")
        ax.plot(it, mean, lw=1.2, label=f"mean of {curves.shape[0]} chains")
        ax.set_xscale("symlog")
        ax.set_xlabel("iteration")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{args.domain} decision tree using {args.proposal}: per-iteration test accuracy")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\nplot written to {args.plot}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
