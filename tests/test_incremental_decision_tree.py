import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

from discretesampling.base.algorithms import DiscreteVariableSMC, DiscreteVariableMCMC
from discretesampling.base.executor import Executor
from discretesampling.base.random import RNG
from discretesampling.base.util import pad, restore
from discretesampling.domain import incremental_decision_tree as idt
from discretesampling.domain.incremental_decision_tree.moves import (
    apply_subtree_proposal, change_admissible, draw_subtree, evaluate_subtree_move,
    make_context, select_move, subtree_admissible, subtree_node_data)


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


@pytest.fixture(scope="module")
def target(problem):
    return idt.IncrementalTreeTarget(problem)


def walk(proposal, problem, seed=0, steps=250, record=False, probe=None):
    """
    Drive a proposal as a chain of its own, collecting the accepted moves.

    `probe(proposal, x, x_prime)` is called the moment a move is accepted, and
    its results are returned alongside. That timing is not incidental: the
    correction a proposal records lives on the particle only until the next
    call involving it -- a later stay overwrites it with the identity's zeros,
    which is exactly what SMC wants and what a test reading it afterwards
    would trip over.
    """
    proposal.record_diagnostics = record
    rng = RNG(seed)
    x = idt.IncrementalTree.stump(problem)
    moves, probes = [], []
    for _ in range(steps):
        x_prime = proposal.sample(x, rng)
        if x_prime is not x:
            moves.append((x, x_prime))
            if probe is not None:
                probes.append(probe(proposal, x, x_prime))
        x = x_prime
    return (x, moves, probes) if probe is not None else (x, moves)


# --------------------------------------------------------------------------- #
# tier 1: the tree itself
# --------------------------------------------------------------------------- #

def test_stump_holds_every_row(problem):
    stump = idt.IncrementalTree.stump(problem)
    assert stump.leafs == [0]
    assert len(stump.leaf_idx[0]) == problem.n_rows
    assert stump.counts[0].sum() == problem.n_rows


def test_stumps_share_their_index_array(problem):
    a, b = idt.IncrementalTree.stump(problem), idt.IncrementalTree.stump(problem)
    # N particles must not each hold a private copy of every row index.
    assert a.leaf_idx[0] is b.leaf_idx[0]


def test_incremental_moves_match_a_rebuild_from_rows(problem):
    """Leaf membership after incremental moves == routing the data from scratch."""
    final, moves = walk(idt.IncrementalTreeProposal(), problem, seed=9, steps=400)
    assert len(moves) > 20
    for _, x in moves:
        fresh = idt.IncrementalTree.from_rows(problem, x.tree)
        assert set(fresh.leaf_idx) == set(x.leaf_idx)
        for leaf in fresh.leaf_idx:
            assert np.array_equal(np.sort(fresh.leaf_idx[leaf]),
                                  np.sort(x.leaf_idx[leaf]))
            assert np.array_equal(fresh.counts[leaf], x.counts[leaf])


def test_context_updates_match_a_fresh_walk(problem):
    """The incrementally maintained context == make_context on the result."""
    rng = RNG(3)
    x = idt.IncrementalTree.stump(problem)
    checked = 0
    for _ in range(400):
        root, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            continue
        idx = subtree_node_data(x, ctx, node)
        pc, prop_move = evaluate_subtree_move(x, ctx, move, node, idx, rng)
        if pc <= idt.BARRED:
            continue
        x2 = x.deep_copy()
        ctx2 = make_context(x2, root)
        apply_subtree_proposal(x2, ctx2, move, prop_move)
        fresh = make_context(x2, root)
        assert sorted(ctx2.nodes) == sorted(fresh.nodes)
        assert sorted(ctx2.terminal_nodes) == sorted(fresh.terminal_nodes)
        assert sorted(ctx2.leaves) == sorted(fresh.leaves)
        assert sorted(ctx2.leaf_idx) == sorted(fresh.leaf_idx)
        checked += 1
        if subtree_admissible(x2, root) and len(x2.tree) < 6:
            x = x2
    assert checked > 50


def test_change_admissible_agrees_with_applying_the_change(problem):
    rng = RNG(17)
    x = idt.IncrementalTree.stump(problem)
    checked = 0
    for _ in range(500):
        root, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            continue
        idx = subtree_node_data(x, ctx, node)
        pc, prop_move = evaluate_subtree_move(x, ctx, move, node, idx, rng)
        if pc <= idt.BARRED:
            continue
        x2 = x.deep_copy()
        ctx2 = make_context(x2, root)
        apply_subtree_proposal(x2, ctx2, move, prop_move)
        if move == "change":
            predicted = change_admissible(x, node, prop_move['feat'],
                                          prop_move['thr'], idx)
            assert predicted == subtree_admissible(x2, root)
            checked += 1
        if subtree_admissible(x2, root) and len(x2.tree) < 6:
            x = x2
    assert checked > 20


def test_encode_decode_round_trip(problem):
    x, _ = walk(idt.IncrementalTreeProposal(), problem, seed=5, steps=300)
    assert len(x.tree) > 0
    decoded = idt.IncrementalTree.decode(idt.IncrementalTree.encode(x), x)
    assert decoded == x
    t = idt.IncrementalTreeTarget(problem)
    assert t.eval(decoded) == pytest.approx(t.eval(x), abs=1e-9)


def test_pad_restore_round_trip(problem):
    """The MPI redistribution path: particles of different sizes, padded together."""
    particles = [idt.IncrementalTree.stump(problem)]
    x, moves = walk(idt.IncrementalTreeProposal(), problem, seed=6, steps=300)
    particles += [m[1] for m in moves[:5]] + [x]
    restored = restore(pad(particles, Executor()), particles)
    assert len(restored) == len(particles)
    for original, copy_ in zip(particles, restored):
        assert original == copy_


# --------------------------------------------------------------------------- #
# tier 1: the correction terms
# --------------------------------------------------------------------------- #

def test_local_move_ratio_equals_the_global_target_ratio(problem, target):
    """
    The single most load-bearing identity in the domain.

    At ss_prop = 1.0 the delayed-acceptance screen is an exact Metropolis test
    against the full target, so its recorded correction must cancel the target
    ratio the SMC reweight computes, leaving nothing but the subtree-root term.
    That can only hold if eval_move's local before/after densities agree
    exactly with the whole-tree target -- i.e. if every leaf, prior and split
    density that the move does not touch really does cancel.
    """
    def weight_update_and_root_term(proposal, x, x_prime):
        return (target.eval(x_prime) - target.eval(x)
                + proposal.eval(x_prime, x) - proposal.eval(x, x_prime),
                x_prime._smc_diag['root_term'])

    proposal = idt.DAProposal(target, ss_prop=1.0, min_data=1)
    _, moves, probes = walk(proposal, problem, seed=3, steps=400, record=True,
                            probe=weight_update_and_root_term)
    assert len(moves) > 20
    for delta_w, root_term in probes:
        assert delta_w == pytest.approx(root_term, abs=1e-9)


def test_grow_and_its_reverse_prune_price_the_same_transition(problem):
    """
    log q(x -> x') for a grow and log q(x' -> x) for the prune that undoes it
    must cancel: pc(grow) + pc(reverse prune) == 0.

    This is what pins the terminal-node bookkeeping. Splitting a node makes it
    terminal but can also stop its parent being terminal, so the reverse
    prune's pool does not simply grow by one, and getting that wrong
    under-weights every grow by |T|/(|T|+1).
    """
    rng = RNG(4)
    x = idt.IncrementalTree.stump(problem)
    pairs = 0
    for _ in range(1200):
        root, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            continue
        idx = subtree_node_data(x, ctx, node)
        pc, prop_move = evaluate_subtree_move(x, ctx, move, node, idx, rng)
        if pc <= idt.BARRED:
            continue
        x2 = x.deep_copy()
        ctx2 = make_context(x2, root)
        apply_subtree_proposal(x2, ctx2, move, prop_move)
        # Only grow -> prune: the reverse of a prune is a grow, and
        # evaluate_subtree_move draws a fresh feature and threshold for a grow
        # instead of the ones that were pruned, so it cannot price that
        # direction of the same transition.
        if move == "grow":
            after = make_context(x2, root)
            idx2 = subtree_node_data(x2, after, node)
            pc_back, _ = evaluate_subtree_move(x2, after, "prune", node, idx2, rng)
            if pc_back > idt.BARRED:
                assert pc + pc_back == pytest.approx(0.0, abs=1e-9)
                pairs += 1
        if subtree_admissible(x2, root) and len(x2.tree) < 8:
            x = x2
    assert pairs > 30


def test_hints_with_one_block_is_delayed_acceptance(problem, target):
    """
    At ss_prop = 1.0 both reduce to a single block over the whole subtree, and
    they draw from the RNG in the same order, so the particle streams must be
    identical -- not merely similar.
    """
    da, _ = walk(idt.DAProposal(target, ss_prop=1.0, min_data=1), problem,
                 seed=7, steps=300)
    hints, _ = walk(idt.HINTSProposal(target, ss_prop=1.0, min_data=1), problem,
                    seed=7, steps=300)
    assert da == hints


@pytest.mark.parametrize("make", [
    lambda t: idt.IncrementalTreeProposal(),
    lambda t: idt.DAProposal(t, ss_prop=0.25, min_data=20),
    lambda t: idt.HINTSProposal(t, ss_prop=0.25, min_data=20),
])
def test_a_stay_is_the_same_object_and_costs_nothing(problem, target, make):
    proposal = make(target)
    rng = RNG(11)
    x = idt.IncrementalTree.stump(problem)
    stays = 0
    for _ in range(300):
        x_prime = proposal.sample(x, rng)
        if x_prime is x:
            stays += 1
            # both halves zero, so the weight update is exactly zero, and the
            # target's memo hits rather than re-evaluating the full data
            assert proposal.eval(x, x_prime) == 0.0
            assert proposal.eval(x_prime, x) == 0.0
        x = x_prime
    assert stays > 0


@pytest.mark.parametrize("make", [
    lambda t: idt.IncrementalTreeProposal(),
    lambda t: idt.DAProposal(t, ss_prop=0.25, min_data=20),
    lambda t: idt.HINTSProposal(t, ss_prop=0.25, min_data=20),
])
def test_sampling_never_mutates_the_particle_it_was_given(problem, target, make):
    """
    After a resample the serial executor hands the same object to several
    slots, so a proposal that mutated its input would corrupt the others.
    """
    proposal = make(target)
    rng = RNG(2)
    x = idt.IncrementalTree.stump(problem)
    for _ in range(300):
        before_rows = idt.IncrementalTree.encode(x)
        before_leaves = {k: v.copy() for k, v in x.leaf_idx.items()}
        x_prime = proposal.sample(x, rng)
        assert np.array_equal(before_rows, idt.IncrementalTree.encode(x))
        assert set(before_leaves) == set(x.leaf_idx)
        for leaf, rows in before_leaves.items():
            assert np.array_equal(rows, x.leaf_idx[leaf])
        x = x_prime


def test_aliased_particles_get_their_own_corrections(problem, target):
    """
    Two slots holding one object -- what resampling produces -- one screened
    out and one moved. Each must read back its own correction.
    """
    proposal = idt.DAProposal(target, ss_prop=0.25, min_data=20)
    shared, _ = walk(idt.IncrementalTreeProposal(), problem, seed=1, steps=120)
    moved = stayed = None
    for seed in range(60):
        rng = RNG(seed)
        result = proposal.sample(shared, rng)
        if result is shared and stayed is None:
            stayed = result
        elif result is not shared and moved is None:
            moved = result
        if moved is not None and stayed is not None:
            break
    assert moved is not None and stayed is not None
    # the stay wrote (shared, 0, 0) onto the shared object; the move recorded
    # its own terms on a fresh object naming the same source
    assert proposal.eval(shared, stayed) == 0.0
    assert moved._smc_terms[0] is shared
    assert proposal.eval(shared, moved) != 0.0 or proposal.eval(moved, shared) != 0.0


def test_eval_refuses_a_transition_it_never_proposed(problem, target):
    """
    The guard that makes use_optimal_L fail loudly. The optimal L-kernel asks
    for eval on every pair of particles, and a path-dependent correction has no
    value to offer for a pair that was never proposed.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.25, min_data=20)
    a, _ = walk(idt.IncrementalTreeProposal(), problem, seed=1, steps=50)
    b, _ = walk(idt.IncrementalTreeProposal(), problem, seed=2, steps=50)
    with pytest.raises(ValueError, match="path-dependent"):
        proposal.eval(a, b)


def test_target_memo_is_exact_and_dropped_by_copies(problem, target):
    x, _ = walk(idt.IncrementalTreeProposal(), problem, seed=8, steps=200)
    first = target.eval(x)
    before = target.n_full_evals
    assert target.eval(x) == first
    assert target.n_full_evals == before          # served from the memo

    copy_ = x.deep_copy()
    assert not hasattr(copy_, '_log_target')      # a copy exists to be mutated
    assert target.eval(copy_) == pytest.approx(first, abs=1e-12)


def test_out_of_support_trees_are_never_returned(problem, target):
    """Every particle a proposal hands back must carry finite target mass."""
    for make in (lambda: idt.IncrementalTreeProposal(),
                 lambda: idt.DAProposal(target, ss_prop=0.25, min_data=20),
                 lambda: idt.HINTSProposal(target, ss_prop=0.25, min_data=20)):
        _, moves = walk(make(), problem, seed=13, steps=300)
        for _, x in moves:
            assert min(len(rows) for rows in x.leaf_idx.values()) >= problem.min_samples_leaf
            assert target.eval(x) > idt.BARRED


def test_hints_keeps_every_intermediate_state_admissible(problem, target):
    """
    A regression guard for the one bug in this domain that no identity test
    can see: a sweep that passes through a state with a starved leaf. From
    there a prune can merge the starved leaf away, while the grow that would
    undo it is barred -- a move with forward probability and no reverse path,
    which biases the sampled distribution by a couple of percent and nothing
    else.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.2, min_data=15)
    proposal.validate_intermediate = True
    _, moves = walk(proposal, problem, seed=21, steps=400)
    assert len(moves) > 20


# --------------------------------------------------------------------------- #
# tier 1: the samplers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["mh", "da", "hints"])
def test_smc_returns_a_usable_sample(problem, target, name):
    """
    The SMC sampler must return a set of particles that are all admissible and have finite target mass.
    """
    proposals = {"mh": lambda: idt.IncrementalTreeProposal(),
                 "da": lambda: idt.DAProposal(target, ss_prop=0.25, min_data=20),
                 "hints": lambda: idt.HINTSProposal(target, ss_prop=0.25, min_data=20)}
    proposal = proposals[name]()
    smc = DiscreteVariableSMC(idt.IncrementalTree, target,
                              idt.IncrementalTreeInitialProposal(problem),
                              proposal=proposal, Lkernel=proposal.lkernel())
    particles = smc.sample(25, 32, seed=1, verbose=False)

    assert len(particles) == 32
    logprobs = np.array([target.eval(p) for p in particles])
    assert np.all(np.isfinite(logprobs))
    assert np.all(logprobs > idt.BARRED)
    # 25 iterations from a stump: the set has moved and has not collapsed
    sizes = idt.tree_sizes(particles)
    assert sizes.min() > 0
    assert len(set(sizes.tolist())) > 1


def test_smc_default_lkernel_matches_the_explicit_one(problem, target):
    """
    Passing Lkernel= is the honest spelling, but the default path has to give
    the same answer: eval works out the direction from the stash either way.
    """
    def run(explicit):
        proposal = idt.DAProposal(target, ss_prop=0.25, min_data=20)
        smc = DiscreteVariableSMC(idt.IncrementalTree, target,
                                  idt.IncrementalTreeInitialProposal(problem),
                                  proposal=proposal,
                                  Lkernel=proposal.lkernel() if explicit else None)
        return smc.sample(20, 16, seed=4, verbose=False)

    a_particles = run(True)
    b_particles = run(False)
    assert all(p == q for p, q in zip(a_particles, b_particles))


def test_mcmc_drives_the_same_proposals(problem, target):
    """
    The base MCMC sampler needs no adapting: eval answers both directions, so
    the ratio it forms is the Metropolis-Hastings one. With the DA proposal
    that composite is exactly delayed-acceptance MCMC.
    """
    proposal = idt.DAProposal(target, ss_prop=0.25, min_data=20)
    chain = DiscreteVariableMCMC(idt.IncrementalTree, target,
                                 idt.IncrementalTreeInitialProposal(problem),
                                 proposal=proposal).sample(200, seed=0, verbose=False)
    assert len(chain) == 200
    assert all(target.eval(x) > idt.BARRED for x in chain)


def test_prediction_and_accuracy(problem, target):
    x, _ = walk(idt.IncrementalTreeProposal(), problem, seed=5, steps=200)
    probs = idt.predict_proba(x, problem.X)
    assert probs.shape == (problem.n_rows, problem.num_classes)
    assert np.allclose(probs.sum(axis=1), 1.0)

    labels = idt.predict([x, x], problem.X, weights=[0.3, 0.7])
    assert labels.shape == (problem.n_rows,)
    # a tree fitted on this data beats the majority-class rate
    assert idt.accuracy(problem.y, labels) > 0.5


# --------------------------------------------------------------------------- #
# tier 2: does it sample the right distribution?
# --------------------------------------------------------------------------- #

def test_subsampled_kernels_sample_the_same_posterior():
    """
    The three mutations, driven as MCMC chains so that the comparison is of
    the kernels and not of SMC's finite-particle behaviour, must agree on the
    posterior over tree size.

    Deliberately a small, weakly-identified problem: the tree space has to be
    small enough that 12k iterations actually mixes, or the seed-to-seed
    spread swamps the effect being tested. On a strongly-identified dataset
    the chains get stuck in different modes and this comparison says nothing.
    """
    problem = make_problem(n=300, d=4, seed=0, lam=1.0,
                           min_samples_leaf=20, max_tree_size=3)
    problem.y = np.random.default_rng(5).integers(0, 2, size=problem.n_rows)
    problem._root_counts = None
    target = idt.IncrementalTreeTarget(problem)
    init = idt.IncrementalTreeInitialProposal(problem)

    # Calibrated, not guessed: with the admissibility check on the sweep
    # removed, this separates at 5+ sigma on P(nodes=0) and P(nodes=1), while
    # the correct kernels sit inside 1 sigma. Fewer seeds or shorter chains and
    # a 2% bias walks straight through.
    def pmf(make, seeds=8, iters=25000, burn=4000):
        out = []
        for seed in range(seeds):
            chain = DiscreteVariableMCMC(
                idt.IncrementalTree, target, init, proposal=make()
            ).sample(iters, seed=seed, verbose=False)[burn:]
            sizes = idt.tree_sizes(chain)
            out.append([float(np.mean(sizes == k)) for k in range(3)])
        return np.array(out)

    reference = pmf(lambda: idt.IncrementalTreeProposal())
    for name, make in [("DA", lambda: idt.DAProposal(target, ss_prop=0.25, min_data=20)),
                       ("HINTS", lambda: idt.HINTSProposal(target, ss_prop=0.25,
                                                           min_data=20))]:
        got = pmf(make)
        for k in range(3):
            diff = got[:, k].mean() - reference[:, k].mean()
            se = np.sqrt(got[:, k].var(ddof=1) / len(got)
                         + reference[:, k].var(ddof=1) / len(reference))
            assert abs(diff) < 3.5 * se + 0.002, (
                f"{name} disagrees with the full-data kernel on P(nodes={k}): "
                f"{got[:, k].mean():.4f} vs {reference[:, k].mean():.4f} "
                f"({diff / max(se, 1e-12):+.1f} sigma)")


def _exact_log_dm(problem, C):
    """
    Dirichlet-Multinomial log-density, vectorised over the rows of C.
    Checked against the target's own scalar log_dm below, so the reference
    stays tied to the implementation it is grading.
    """
    a0, alpha, K = problem._a0, problem.alpha, problem.num_classes
    return (gammaln(a0) - gammaln(a0 + C.sum(axis=1))
            + gammaln(C + alpha).sum(axis=1) - K * gammaln(alpha))


def exact_p_stump(problem, target, grid=50_001):
    """
    The exact P(0 internal nodes) for a problem capped at max_tree_size=1.

    With that cap the model is either the root-only tree or a single split
    (feature f, threshold t), so the normalising constant is a 1-D integral
    per feature rather than a sum over an unbounded tree space:

        pi(0) = exp(log_prior(0, 1) + log_dm(root counts))
        pi(1) = exp(log_prior(1, 2) + lp_feats)
                * sum_f  Integral p(t) exp(log_dm(left) + log_dm(right)) dt

    exp(problem.lp_vals[f]) is exactly the uniform density of t over that
    feature's range, so the integral against it is a plain average over an
    evenly spaced grid. Sorting the rows by the feature turns every threshold
    into a prefix cut, so all grid points are evaluated in one pass.
    """
    lp0 = target.log_prior(0, 1) + target.log_dm(problem.root_counts())

    K, y = problem.num_classes, problem.y
    per_feature = []
    for f in range(problem.n_features):
        v_min, v_max = problem.vals[f]
        thr = np.linspace(v_min, v_max, grid)
        col = np.asarray(problem.X[:, f])
        order = np.argsort(col, kind="stable")
        ys, cols = y[order], col[order]
        cumulative = np.zeros((len(ys) + 1, K), dtype=np.int64)
        for k in range(K):
            cumulative[1:, k] = np.cumsum(ys == k)
        # a row goes left when its feature value is < thr
        left = cumulative[np.searchsorted(cols, thr, side="left")]
        right = cumulative[-1] - left
        ll = _exact_log_dm(problem, left) + _exact_log_dm(problem, right)
        per_feature.append(logsumexp(ll) - np.log(grid))

    lp1 = target.log_prior(1, 2) + problem.lp_feats + logsumexp(per_feature)
    return float(np.exp(lp0 - logsumexp([lp0, lp1])))


def test_exact_log_dm_reference_matches_the_target():
    """The vectorised reference must agree with the target's own log_dm."""
    problem = make_problem(n=40, d=2, seed=3, min_samples_leaf=0, max_tree_size=1)
    target = idt.IncrementalTreeTarget(problem)
    counts = np.array([[0, 0], [1, 0], [3, 5], [12, 1], [7, 7], [20, 0]])
    assert np.allclose(_exact_log_dm(problem, counts),
                       [target.log_dm(c) for c in counts])


@pytest.mark.parametrize("name", ["MH", "DA", "HINTS"])
def test_kernels_match_the_exact_posterior(name):
    """
    Every kernel must reproduce the exact posterior over model size, not merely
    agree with each other.

    test_subsampled_kernels_sample_the_same_posterior uses the MH kernel as its
    reference, so a bias shared by all three -- or one in MH alone -- passes it
    unnoticed. This grades them against a posterior computed in closed form.

    Weakly identified on purpose: y carries no signal and lam is tuned so the
    two models sit near 0.59/0.41. Where one model dominates, P(0) pins to 0 or
    1 and the comparison has no power to detect anything.
    """
    problem = make_problem(n=60, d=2, seed=0, lam=1.0,
                           min_samples_leaf=0, max_tree_size=1)
    problem.y = np.random.default_rng(0).integers(
        0, 2, size=problem.n_rows).astype(np.int64)
    problem._root_counts = None

    target = idt.IncrementalTreeTarget(problem)
    init = idt.IncrementalTreeInitialProposal(problem)
    exact = exact_p_stump(problem, target)
    assert 0.2 < exact < 0.8, f"reference problem lost its power: P(0)={exact}"

    make = {
        "MH": lambda: idt.IncrementalTreeProposal(),
        "DA": lambda: idt.DAProposal(target, ss_prop=0.25, min_data=20),
        "HINTS": lambda: idt.HINTSProposal(target, ss_prop=0.25, min_data=20),
    }[name]

    # Calibrated, not guessed. Against the unmodified kernels this sits at
    # ~1.6 sigma; dropping the threshold density from the grow correction shows
    # up at 216 sigma, and over-weighting grow by 15% -- the |T|/(|T|+1) slip
    # evaluate_subtree_move warns about -- at 6.7 sigma. Halve either the seeds
    # or the iterations and that second one walks through.
    estimates = []
    for seed in range(8):
        chain = DiscreteVariableMCMC(
            idt.IncrementalTree, target, init, proposal=make()
        ).sample(40_000, seed=seed, verbose=False)[5_000:]
        estimates.append(float(np.mean(idt.tree_sizes(chain) == 0)))

    estimates = np.array(estimates)
    se = estimates.std(ddof=1) / np.sqrt(len(estimates))
    diff = estimates.mean() - exact
    assert abs(diff) < 4.0 * se + 0.003, (
        f"{name} disagrees with the exact posterior on P(0 nodes): "
        f"{estimates.mean():.4f} vs {exact:.4f} "
        f"({diff / max(se, 1e-12):+.1f} sigma)")


def test_non_uniform_threshold_proposal_keeps_the_target():
    """
    A threshold proposal that is not the prior must still sample the prior's
    posterior.

    This is the regression guard for splitting lp_thr_proposal (proposal
    density) out of lp_vals (threshold prior). While the reverse corrections in
    evaluate_subtree_move read lp_vals, any non-uniform proposal scored the
    forward draw under one density and the reverse under another, which breaks
    detailed balance while *raising* the acceptance rate -- so it looks like
    better mixing rather than a bug. The exact reference below depends only on
    the prior, so it is unmoved by threshold_proposal and catches exactly that.
    """
    problem = make_problem(n=60, d=2, seed=0, lam=1.0, min_samples_leaf=0,
                           max_tree_size=1, threshold_proposal="data")
    problem.y = np.random.default_rng(0).integers(
        0, 2, size=problem.n_rows).astype(np.int64)
    problem._root_counts = None

    target = idt.IncrementalTreeTarget(problem)
    init = idt.IncrementalTreeInitialProposal(problem)
    exact = exact_p_stump(problem, target)

    estimates = []
    for seed in range(8):
        chain = DiscreteVariableMCMC(
            idt.IncrementalTree, target, init,
            proposal=idt.IncrementalTreeProposal()
        ).sample(40_000, seed=seed, verbose=False)[5_000:]
        estimates.append(float(np.mean(idt.tree_sizes(chain) == 0)))

    estimates = np.array(estimates)
    se = estimates.std(ddof=1) / np.sqrt(len(estimates))
    diff = estimates.mean() - exact
    assert abs(diff) < 4.0 * se + 0.003, (
        "the 'data' threshold proposal does not target the prior's posterior: "
        f"{estimates.mean():.4f} vs {exact:.4f} "
        f"({diff / max(se, 1e-12):+.1f} sigma)")


def test_threshold_proposal_density_round_trips():
    """random_threshold's returned density must be what lp_thr_proposal scores."""
    for tp in ("uniform", "data"):
        problem = make_problem(n=80, d=3, seed=1, threshold_proposal=tp)
        rng = RNG(0)
        for _ in range(200):
            feat = problem.random_feature(rng)
            thr, lp = problem.random_threshold(feat, rng)
            assert np.isclose(lp, problem.lp_thr_proposal(feat, thr)), (
                f"{tp}: draw scored {lp} but lp_thr_proposal says "
                f"{problem.lp_thr_proposal(feat, thr)}")
            v_min, v_max = problem.vals[feat]
            assert v_min <= thr <= v_max


def test_unknown_threshold_proposal_is_rejected():
    with pytest.raises(ValueError, match="threshold_proposal"):
        make_problem(threshold_proposal="kde")
