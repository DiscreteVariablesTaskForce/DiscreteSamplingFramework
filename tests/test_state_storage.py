"""
Tests for the flat state storage the samplers record into.

The premise of storing states and evaluating them later is that a stored state
is the state -- so most of what is checked here is that a metric read off a
stored run equals the metric the live tree would have given, for both samplers
and every proposal. The rest guards the two things that make the storage small:
identity-based deduplication, and evaluating a deduplicated ensemble.
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
from discretesampling.domain import incremental_decision_tree as idt
from discretesampling.domain.incremental_decision_tree.states import (
    StateRecorder, series_from_arrays)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def make_problem(n=300, d=4, seed=0, **kwargs):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, d))
    y = (X[:, 0] + 0.9 * rs.normal(size=n) > 0).astype(np.int64)
    opts = dict(lam=3.0, min_samples_leaf=10, max_tree_size=8)
    opts.update(kwargs)
    return idt.IncrementalTreeProblem(X, y, **opts)


@pytest.fixture(scope="module")
def problem():
    return make_problem()


def build_proposal(problem, name):
    target = idt.IncrementalTreeTarget(problem)
    if name == "MH":
        return target, idt.IncrementalTreeProposal()
    if name == "DA":
        return target, idt.DAProposal(target, ss_prop=0.3, min_data=20)
    return target, idt.HINTSProposal(target, ss_prop=0.3, min_data=20)


def grown(problem, splits):
    """A tree with `splits` applied, for a state that is not a stump."""
    tree = idt.IncrementalTree.stump(problem)
    for leaf, feat, thr in splits:
        tree = tree.deep_copy().grow(leaf, feat, thr)
    return tree


# --------------------------------------------------------------------------- #
# a stored state is the state
# --------------------------------------------------------------------------- #

def test_stored_stump_matches_live(problem):
    stump = idt.IncrementalTree.stump(problem)
    rec = StateRecorder(problem.num_classes, problem.alpha)
    rec.record([stump])
    stored = rec.series().state(0)

    assert len(stored.tree) == 0
    assert np.array_equal(stored.route(problem.X), stump.route(problem.X))
    assert np.allclose(idt.predict_proba(stored, problem.X),
                       idt.predict_proba(stump, problem.X))


def test_stored_tree_matches_live(problem):
    tree = grown(problem, [(0, 0, 0.0), (1, 1, -0.25), (2, 2, 0.5)])
    rec = StateRecorder(problem.num_classes, problem.alpha)
    rec.record([tree])
    stored = rec.series().state(0)

    assert len(stored.tree) == len(tree.tree)
    assert sorted(stored.counts.keys()) == sorted(tree.counts.keys())
    for leaf in tree.counts:
        assert np.array_equal(stored.counts[leaf], tree.counts[leaf])
    # Routing is the test that actually matters: it is what a float32 threshold
    # would break, and it is what every metric goes through.
    assert np.array_equal(stored.route(problem.X), tree.route(problem.X))
    assert np.allclose(idt.predict_proba(stored, problem.X),
                       idt.predict_proba(tree, problem.X))


@pytest.mark.parametrize("proposal_name", ["MH", "DA", "HINTS"])
def test_mcmc_stored_metrics_match_inline(problem, proposal_name):
    """Every metric off the stored chain equals the one the live tree gave."""
    target, proposal = build_proposal(problem, proposal_name)
    mcmc = DiscreteVariableMCMC(idt.IncrementalTree, target,
                                idt.IncrementalTreeInitialProposal(problem),
                                proposal=proposal)
    rec = StateRecorder(problem.num_classes, problem.alpha)
    inline = []

    def callback(i, current, accepted):
        rec.record([current], iteration=i)
        inline.append(idt.evaluate([current], problem.X, problem.y,
                                   num_classes=problem.num_classes))

    mcmc.sample(200, seed=4, verbose=False, callback=callback)

    series = rec.series()
    assert len(series) == len(inline) == 200
    for i, expected in enumerate(inline):
        slots, weights = series.ensemble(i)
        got = idt.evaluate([series.state(s) for s in slots], problem.X, problem.y,
                           weights=weights, num_classes=series.num_classes)
        for key, value in expected.items():
            assert np.allclose(got[key], value), f"{proposal_name} iter {i}: {key}"


@pytest.mark.parametrize("proposal_name", ["MH", "DA", "HINTS"])
def test_smc_stored_metrics_match_inline(problem, proposal_name):
    """
    The weighted ensemble estimator too: the weights are stored unnormalised
    and normalised on the way out, so this checks that round trip as well as
    the states.
    """
    target, proposal = build_proposal(problem, proposal_name)
    smc = DiscreteVariableSMC(idt.IncrementalTree, target,
                              idt.IncrementalTreeInitialProposal(problem),
                              proposal=proposal, Lkernel=proposal.lkernel())
    rec = StateRecorder(problem.num_classes, problem.alpha)
    inline = []

    def callback(t, particles, logWeights, neff, resampled):
        rec.record(particles, iteration=t, log_weights=logWeights)
        w = np.exp(logWeights - logsumexp(logWeights))
        inline.append(idt.evaluate(particles, problem.X, problem.y, weights=w,
                                   num_classes=problem.num_classes))

    smc.sample(10, 20, seed=2, verbose=False, callback=callback)

    series = rec.series()
    assert len(series) == 10 and series.n_particles == 20
    for i, expected in enumerate(inline):
        slots, weights = series.ensemble(i)
        got = idt.evaluate([series.state(s) for s in slots], problem.X, problem.y,
                           weights=weights, num_classes=series.num_classes)
        for key, value in expected.items():
            assert np.allclose(got[key], value), f"{proposal_name} step {i}: {key}"


# --------------------------------------------------------------------------- #
# deduplication
# --------------------------------------------------------------------------- #

def test_repeated_state_stored_once(problem):
    """A chain that rejects sits on one object, and stores one state."""
    stump = idt.IncrementalTree.stump(problem)
    other = grown(problem, [(0, 0, 0.0)])
    rec = StateRecorder(problem.num_classes, problem.alpha)
    for state in (stump, stump, stump, other, other, stump):
        rec.record([state])

    series = rec.series()
    assert len(series) == 6
    # The last stump is a third stored state: the dedup window is the current
    # record and the one before it, so a state the chain returns to after
    # leaving it is stored again rather than matched.
    assert series.n_stored == 3
    assert [int(series.slots(i)[0]) for i in range(6)] == [0, 0, 0, 1, 1, 2]


def test_aliased_particles_stored_once(problem):
    """Resampling puts one object in many slots; it costs one stored state."""
    stump = idt.IncrementalTree.stump(problem)
    other = grown(problem, [(0, 1, 0.1)])
    rec = StateRecorder(problem.num_classes, problem.alpha)
    rec.record([stump, stump, other, stump])

    series = rec.series()
    assert series.n_stored == 2
    assert [int(s) for s in series.slots(0)] == [0, 0, 1, 0]


def test_ensemble_dedup_matches_full_ensemble(problem):
    """
    Evaluating the deduplicated ensemble is exact, not an approximation: the
    combined weights reproduce the full particle-by-particle average.
    """
    stump = idt.IncrementalTree.stump(problem)
    other = grown(problem, [(0, 0, 0.2)])
    particles = [stump, other, stump, other, stump]
    logw = np.log([0.1, 0.4, 0.2, 0.2, 0.1])

    rec = StateRecorder(problem.num_classes, problem.alpha)
    rec.record(particles, log_weights=logw)
    series = rec.series()

    slots, weights = series.ensemble(0)
    assert len(slots) == 2 and np.isclose(weights.sum(), 1.0)
    deduped = idt.ensemble_predict_proba([series.state(s) for s in slots],
                                         problem.X, weights)
    full = idt.ensemble_predict_proba(particles, problem.X, series.weights(0))
    assert np.allclose(deduped, full)


def test_changing_ensemble_size_is_refused(problem):
    stump = idt.IncrementalTree.stump(problem)
    rec = StateRecorder(problem.num_classes, problem.alpha)
    rec.record([stump, stump])
    with pytest.raises(ValueError, match="ensemble size changed"):
        rec.record([stump])


# --------------------------------------------------------------------------- #
# the arrays, and the round trip through them
# --------------------------------------------------------------------------- #

def test_series_survives_the_array_round_trip(problem):
    """
    What a results file holds is arrays(); rebuilding from them has to give
    back the same series, since that is the only route an evaluation takes.
    """
    target, proposal = build_proposal(problem, "DA")
    smc = DiscreteVariableSMC(idt.IncrementalTree, target,
                              idt.IncrementalTreeInitialProposal(problem),
                              proposal=proposal, Lkernel=proposal.lkernel())
    rec = StateRecorder(problem.num_classes, problem.alpha)

    def callback(t, particles, logWeights, neff, resampled):
        rec.record(particles, iteration=2 * t, log_weights=logWeights)

    smc.sample(6, 12, seed=7, verbose=False, callback=callback)

    original = rec.series()
    # Through numpy and back, as h5py would hand them over.
    arrays = {k: (np.asarray(v) if np.ndim(v) else v)
              for k, v in rec.arrays().items()}
    restored = series_from_arrays(arrays)

    assert len(restored) == len(original)
    assert restored.n_particles == original.n_particles
    assert restored.n_stored == original.n_stored
    assert restored.num_classes == original.num_classes
    assert restored.alpha == original.alpha
    assert np.array_equal(restored.record_iterations(),
                          original.record_iterations())
    assert np.array_equal(restored.record_iterations(),
                          np.arange(6, dtype=np.int64) * 2)
    for i in range(len(original)):
        assert np.array_equal(restored.slots(i), original.slots(i))
        assert np.allclose(restored.weights(i), original.weights(i))
    for slot in range(original.n_stored):
        before, after = original.state(slot), restored.state(slot)
        assert np.array_equal(np.asarray(before.tree), np.asarray(after.tree))
        assert np.array_equal(before.counts.ids, after.counts.ids)
        assert np.array_equal(before.counts.block, after.counts.block)


def test_series_from_arrays_without_states_is_none():
    assert series_from_arrays({'n_nodes': np.zeros(3)}) is None


def test_unweighted_records_are_uniform(problem):
    """MCMC stores no weights; a record is then the plain average."""
    stump = idt.IncrementalTree.stump(problem)
    rec = StateRecorder(problem.num_classes, problem.alpha)
    rec.record([stump])
    series = rec.series()

    assert series.log_weights is None
    assert series.weights(0) is None
    _, weights = series.ensemble(0)
    assert np.allclose(weights, [1.0])
