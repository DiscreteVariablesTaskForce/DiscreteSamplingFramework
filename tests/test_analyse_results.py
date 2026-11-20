"""
The analysis recomputes each stored tree's log posterior from its stored
splits and leaf counts, since the runs store trees and not their scores. It
must give exactly what the target gave the live tree.
"""
import os
import sys

import numpy as np
from sklearn.datasets import load_wine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "examples", "incremental_decision_tree"))

import analyse_results as ar  # noqa: E402
from discretesampling.base.random import RNG  # noqa: E402
from discretesampling.domain import incremental_decision_tree as idt  # noqa: E402


def test_stored_log_posterior_matches_the_target():
    X, y = load_wine(return_X_y=True)
    problem = idt.IncrementalTreeProblem(X, y, lam=6.0, min_samples_leaf=3)
    target = idt.IncrementalTreeTarget(problem)
    start = idt.IncrementalTreeInitialProposal(problem)
    trees = [start.sample(RNG(seed), target) for seed in range(40)]
    trees.append(idt.IncrementalTreeInitialProposal(problem, n_nodes=0).sample(RNG(0), target))

    recorder = idt.StateRecorder(problem.num_classes)
    for tree in trees:
        recorder.record([tree])
    run = recorder.arrays()

    lp_vals, lp_feats = np.array(problem.lp_vals), problem.lp_feats
    log_post, nodes, root_feat, _, keys = ar.stored_states(run, problem.lam, lp_vals, lp_feats)
    slots = run["state_index"][:, 0]

    np.testing.assert_allclose(log_post[slots], [target.eval(t) for t in trees], rtol=0, atol=1e-8)
    assert list(nodes[slots]) == [len(t.tree) for t in trees]
    assert root_feat[slots][-1] == -1                   # the stump
    assert len(np.unique(keys[slots])) == len({tuple(sorted(map(tuple, t.tree))) for t in trees})


def test_iact_of_an_ar1_series():
    rng = np.random.default_rng(0)
    phi, n = 0.8, 200_000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    assert abs(ar.iact(x) - (1 + phi) / (1 - phi)) < 0.5
    assert np.isnan(ar.iact(np.ones(100)))


def test_rhat_separates_chains_in_different_places():
    rng = np.random.default_rng(1)
    same = [rng.normal(size=2000) for _ in range(4)]
    apart = [rng.normal(loc=3 * i, size=2000) for i in range(4)]
    assert ar.rhat(same) < 1.01
    assert ar.rhat(apart) > 2
