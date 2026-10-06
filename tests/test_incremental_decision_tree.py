import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

from discretesampling.base.algorithms import DiscreteVariableSMC, DiscreteVariableMCMC
from discretesampling.base.executor import Executor
from discretesampling.base.random import RNG
from discretesampling.base.util import pad, restore
from discretesampling.domain import incremental_decision_tree as idt
from discretesampling.domain.incremental_decision_tree import diagnostics as dg
from discretesampling.domain.incremental_decision_tree.moves import (
    apply_subtree_proposal, change_admissible, change_partition, check_grow_split,
    draw_subtree, evaluate_subtree_move, make_context, select_move,
    subtree_admissible, subtree_leaves, subtree_node_data, valid_split)
from discretesampling.domain.incremental_decision_tree.subsampling import (
    RowBlocks, grouped_rows)
from discretesampling.domain.incremental_decision_tree.proposals.hints import (
    BlockRows, NestedBlocks)


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


def largest_tree_visited(problem, seed, steps=400):
    """The biggest tree an MH proposal walk passes through -- its last state
    can be a stump again, with nothing to change or prune."""
    _, moves = walk(idt.IncrementalTreeProposal(), problem, seed=seed, steps=steps)
    return max((after for _, after in moves), key=lambda tree: len(tree.tree))


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


def test_a_change_installed_from_its_partition_is_the_change_routed(problem):
    """FlatHINTS applies a change from the partition its admissibility check built."""
    rng = RNG(21)
    x = largest_tree_visited(problem, seed=5)
    checked = 0
    for _ in range(600):
        _, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move != "change":
            continue
        idx = subtree_node_data(x, ctx, node)
        _, prop_move = evaluate_subtree_move(x, ctx, move, node, idx, rng)
        partition = change_partition(x, node, prop_move['feat'], prop_move['thr'], idx)
        if partition is None:
            continue
        routed, installed = x.deep_copy(), x.deep_copy()
        routed.change(node, prop_move['feat'], prop_move['thr'])
        installed.change(node, prop_move['feat'], prop_move['thr'], partition)
        # Leaf order and row order both matter: later draws index into them.
        assert list(installed.leaf_idx) == list(routed.leaf_idx)
        for leaf in routed.leaf_idx:
            assert np.array_equal(installed.leaf_idx[leaf], routed.leaf_idx[leaf])
            assert np.array_equal(installed.counts[leaf], routed.counts[leaf])
        checked += 1
    assert checked > 20


def test_grow_split_check_agrees_with_splitting_every_row():
    """
    The first rows settle a grow's split only when they already make it valid,
    so the verdict never differs from splitting every row.
    """
    problem = make_problem(n=3000, d=3, seed=4, min_samples_leaf=40)
    rows = problem.all_rows()
    rs = np.random.default_rng(0)
    outcomes = set()
    for _ in range(400):
        feat = int(rs.integers(problem.n_features))
        thr = rs.uniform(*problem.vals[feat])
        idx = rows if rs.random() < 0.5 else rs.choice(
            rows, size=int(rs.integers(50, 3000)), replace=False)
        valid, go_left = check_grow_split(problem, idx, feat, thr)
        full = problem.X[idx, feat] < thr
        assert valid == valid_split(problem, full)
        if go_left is not None:
            assert np.array_equal(go_left, full)
        outcomes.add((valid, go_left is None))
    assert outcomes >= {(True, True), (True, False), (False, False)}


def test_row_blocks_partition_the_rows_under_a_node_as_leaves_change(problem):
    """
    Each block is exactly the rows under the node whose label is its own, so
    the blocks partition those rows -- for fresh splits of any size, and
    however the leaves have changed since the split was drawn.
    """
    rng = RNG(13)
    x = largest_tree_visited(problem, seed=6).deep_copy()
    ctx = make_context(x, 0)
    blocks = RowBlocks(problem.n_rows)
    applied = 0
    for _ in range(80):
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            continue
        num_blocks = int(rng.nprng.integers(2, 8))
        blocks.draw(np.concatenate(list(ctx.leaf_idx.values())), num_blocks, rng)

        leaves = subtree_leaves(x, ctx, node)
        node_rows = subtree_node_data(x, ctx, node)
        which = blocks.labels[node_rows]
        assert np.all((which >= 0) & (which < num_blocks))

        leaf_rows = [ctx.leaf_idx[leaf] for leaf in leaves]
        seen = []
        for j in range(num_blocks):
            subset, leaf_pos = blocks.block_rows(leaf_rows, j)
            assert np.array_equal(np.sort(subset), np.sort(node_rows[which == j]))
            for k, rows in enumerate(leaf_rows):
                assert np.isin(subset[leaf_pos == k], rows).all()
            seen.append(subset)
        # none shared, none dropped
        assert np.array_equal(np.sort(np.concatenate(seen)), np.sort(node_rows))
        pc, prop_move = evaluate_subtree_move(
            x, ctx, move, node, ctx.leaf_idx[node] if move == "grow" else None, rng)
        if pc > idt.BARRED:
            apply_subtree_proposal(x, ctx, move, prop_move)
            applied += 1
    assert applied > 10


def test_flat_hints_draws_a_new_split_every_sweep(problem, target):
    """
    FlatHINTS splits the rows under the subtree root afresh for every sweep
    that has more than one block, and draws nothing for a sweep that has one.
    """
    proposal = idt.FlatHINTSProposal(target, ss_prop=0.2, min_data=15)
    draws = []
    draw = proposal.blocks.draw

    def watched(rows, num_blocks, rng):
        draws.append(num_blocks)
        return draw(rows, num_blocks, rng)

    proposal.blocks.draw = watched
    rng = RNG(5)
    x = idt.IncrementalTree.stump(problem)
    multi = 0
    for _ in range(200):
        before, blocks_before = len(draws), proposal.n_blocks_total
        x = proposal.sample(x, rng)
        num_blocks = proposal.n_blocks_total - blocks_before
        assert len(draws) - before == (1 if num_blocks > 1 else 0)
        if num_blocks > 1:
            assert draws[-1] == num_blocks
            multi += 1
    assert multi > 100


def test_surrogate_scores_the_subset_as_routing_it_would():
    """
    eval_move reads the current tree's side of a move off the leaf each subset
    row sits in, and a change routes only the rows its new split sends across.
    Both have to give exactly the subtree density that routing every subset
    row through the tree before and after the move gives.
    """
    problem = make_problem(n=800, d=5, seed=1, min_samples_leaf=5, max_tree_size=24)
    target = idt.IncrementalTreeTarget(problem)
    x = largest_tree_visited(problem, seed=2, steps=800)
    rng, rs = RNG(8), np.random.default_rng(8)
    K = problem.num_classes

    def subtree_density(tree, node, rows, scale, split_nodes):
        reached = tree.route(problem.X[rows])
        lhood = sum(target.log_dm(np.bincount(problem.y[rows[reached == leaf]], minlength=K) * scale)
                    for leaf in tree._descendant_leaves(node))
        return (lhood + target.sum_split_priors(tree, split_nodes)
                + target.log_prior(len(tree.tree), len(tree.leaf_idx)))

    checked = dict.fromkeys(("grow", "prune", "change"), 0)
    for _ in range(1500):
        root, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            continue
        leaves = subtree_leaves(x, ctx, node)
        rows, leaf_pos = grouped_rows([ctx.leaf_idx[leaf] for leaf in leaves])
        pc, prop_move = evaluate_subtree_move(
            x, ctx, move, node, ctx.leaf_idx[node] if move == "grow" else None, rng)
        keep = rs.random(len(rows)) < 0.3
        if pc <= idt.BARRED or not keep.any():
            continue
        subset, scale = rows[keep], len(rows) / keep.sum()
        v, v_prime = target.eval_move(x, move, node, prop_move, subset, leaf_pos[keep],
                                      leaves, scale)

        x2 = x.deep_copy()
        apply_subtree_proposal(x2, make_context(x2, root), move, prop_move)
        assert v == pytest.approx(subtree_density(
            x, node, subset, scale, [] if move == "grow" else [node]), abs=1e-8)
        assert v_prime == pytest.approx(subtree_density(
            x2, node, subset, scale, [] if move == "prune" else [node]), abs=1e-8)
        checked[move] += 1
    assert min(checked.values()) > 20


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

    On every row under the move node, the screen's surrogate is the target
    itself, so its before/after densities must differ by exactly the
    whole-tree target ratio. That can only hold if eval_subset's local
    densities agree exactly with the whole-tree target -- i.e. if every leaf,
    prior and split density that the move does not touch really does cancel.
    (DA no longer screens on the full data -- with one block it takes a plain
    MH step -- so the identity is checked on eval_subset directly.)
    """
    rng = RNG(3)
    x = idt.IncrementalTree.stump(problem)
    checked = dict.fromkeys(("grow", "prune", "change"), 0)
    for _ in range(1500):
        root, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            continue
        leaves = subtree_leaves(x, ctx, node)
        leaf_rows = [ctx.leaf_idx[leaf] for leaf in leaves]
        pc, prop_move = evaluate_subtree_move(
            x, ctx, move, node, leaf_rows[0] if move == "grow" else None, rng)
        if pc <= idt.BARRED:
            continue
        rows, leaf_pos = grouped_rows(leaf_rows)
        v, v_prime, ok = target.eval_subset(x, move, node, prop_move, rows,
                                            leaf_pos, leaves, len(rows))
        assert ok
        x2 = x.deep_copy()
        apply_subtree_proposal(x2, make_context(x2, root), move, prop_move)
        if not subtree_admissible(x2, root):
            continue
        assert v_prime - v == pytest.approx(target.eval(x2) - target.eval(x), abs=1e-9)
        checked[move] += 1
        if len(x2.tree) < 10:
            x = x2
    assert min(checked.values()) > 20


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


def test_delayed_acceptance_with_one_block_is_metropolis_hastings(problem, target):
    """
    At ss_prop = 1.0 there is a single block over the whole subtree, so there
    is no subsample to screen on and DA skips its screen. It then draws from the
    RNG in the same order as the plain MH proposal, so the particle streams must
    be identical -- not merely similar.
    """
    da, _ = walk(idt.DAProposal(target, ss_prop=1.0, min_data=1), problem,
                 seed=7, steps=300)
    mh, _ = walk(idt.IncrementalTreeProposal(), problem, seed=7, steps=300)
    assert da == mh


@pytest.mark.parametrize("make", [
    lambda t: idt.IncrementalTreeProposal(),
    lambda t: idt.DAProposal(t, ss_prop=0.25, min_data=20),
    lambda t: idt.FlatHINTSProposal(t, ss_prop=0.25, min_data=20),
    lambda t: idt.HINTSProposal(t, ss_prop=0.25, min_data=20),
], ids=["MH", "DA", "FlatHINTS", "HINTS"])
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
    lambda t: idt.FlatHINTSProposal(t, ss_prop=0.25, min_data=20),
    lambda t: idt.HINTSProposal(t, ss_prop=0.25, min_data=20),
], ids=["MH", "DA", "FlatHINTS", "HINTS"])
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
    proposal = idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20)
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
                 lambda: idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20),
                 lambda: idt.HINTSProposal(target, ss_prop=0.25, min_data=20)):
        _, moves = walk(make(), problem, seed=13, steps=300)
        for _, x in moves:
            assert min(len(rows) for rows in x.leaf_idx.values()) >= problem.min_samples_leaf
            assert target.eval(x) > idt.BARRED


def test_hints_blocks_nest_and_partition_their_parent(problem, target):
    """
    The hierarchy's whole premise: a data node's children split its rows between
    them, none shared and none dropped, at every level and whatever the leaves
    look like. Blocks that only nearly nest would leave the path correction
    comparing a node against rows some child never had.
    """
    blocks = NestedBlocks(problem.n_rows, problem.y)
    x = largest_tree_visited(problem, seed=6)
    leaves = x._descendant_leaves(0)
    rows = [x.leaf_idx[leaf] for leaf in leaves]
    units = 27                                  # three levels of three
    blocks.draw(np.concatenate(rows), units, RNG(2))

    def in_range(r, lo, width):
        """Which of `r` lie in the node (lo, width), worked out from the labels
        themselves rather than from the sorted route split() takes."""
        lab = blocks.labels[r]
        return r[(lab >= lo) & (lab < lo + width)]

    rng = RNG(9)
    frontier = [(0, units)]
    checked = 0
    for _ in range(3):
        parents, frontier = frontier, []
        for lo, width in parents:
            children = blocks.ranges(lo, width, 3)
            order = blocks.order(3, rng)
            assert sorted(order.tolist()) == [0, 1, 2]
            assert sum(w for _, w in children) == width

            for r in rows:
                mine = in_range(r, lo, width)
                batched = blocks.split(mine, lo, width, 3, problem.num_classes)
                seen = []
                for (clo, cwidth), (idx, counts) in zip(children, batched):
                    assert np.array_equal(np.sort(idx), np.sort(in_range(r, clo, cwidth)))
                    assert np.array_equal(
                        counts, np.bincount(problem.y[idx],
                                            minlength=problem.num_classes))
                    assert not counts.flags.writeable
                    seen.append(idx)
                # none shared, none dropped: the children are exactly the parent
                assert np.array_equal(np.sort(np.concatenate(seen)), np.sort(mine))
                checked += 1
            frontier.extend(children)
    # every leaf, at the root and at both levels of three below it
    assert checked == len(rows) * (1 + 3 + 9)


def test_hints_draw_splits_the_rows_uniformly_at_random(problem, target):
    """
    With equal=False a draw gives every row under the subtree root one
    uniformly random unit block: every label is a valid block, each block
    holds about 1/U of the rows, and two draws give two different splits.
    """
    n = 12000
    blocks = NestedBlocks(n, np.zeros(n, dtype=np.int64), equal=False)
    rows = np.arange(n)
    for units in (2, 4, 7, 16):
        blocks.draw(rows, units, RNG(units))
        labels = blocks.labels[rows]
        assert labels.min() >= 0 and labels.max() < units
        sizes = np.bincount(labels, minlength=units)
        expected = n / units
        # binomial sizes: well within five standard deviations of n / U
        assert np.all(np.abs(sizes - expected) < 5 * np.sqrt(expected))
        assert blocks.n_units == units

    blocks = NestedBlocks(problem.n_rows, problem.y, equal=False)
    rows = problem.all_rows()[::3].copy()
    blocks.draw(rows, 4, RNG(1))
    first = blocks.labels[rows].copy()
    blocks.draw(rows, 4, RNG(2))
    assert not np.array_equal(blocks.labels[rows], first)


def test_hints_equal_blocks_cut_the_rows_into_equal_shares(problem, target):
    """
    equal=True shuffles and cuts: every row gets a valid block, the blocks'
    sizes differ by at most one, and two draws give two different splits.
    """
    n = 12001
    blocks = NestedBlocks(n, np.zeros(n, dtype=np.int64), equal=True)
    rows = np.arange(n)[::-1].copy()
    for units in (2, 4, 7, 16):
        blocks.draw(rows, units, RNG(units))
        sizes = np.bincount(blocks.labels[rows], minlength=units)
        assert len(sizes) == units
        assert sizes.max() - sizes.min() <= 1
        assert blocks.n_units == units
    first = blocks.labels[rows].copy()
    blocks.draw(rows, 16, RNG(99))
    assert not np.array_equal(blocks.labels[rows], first)


@pytest.mark.parametrize("make", [
    lambda n: RowBlocks(n),
    lambda n: NestedBlocks(n, np.zeros(n, dtype=np.int64)),
], ids=["FlatHINTS", "HINTS"])
def test_equal_blocks_are_the_default_and_their_labels_are_exchangeable(make):
    """
    Both proposals cut the rows into equal blocks by default. When the rows do
    not divide evenly, the cut alone would always put the larger blocks at the
    same labels, so a split and its label-reversed twin would not be equally
    likely -- and FlatHINTS, which visits the blocks in label order, relies on
    them being so. The labels are shuffled after the cut, so every label has to
    hold the larger block about as often as any other.
    """
    n, units, draws = 10, 4, 4000          # sizes 3, 3, 2, 2 in some order
    blocks = make(n)
    rows = np.arange(n)
    rng = RNG(5)
    larger = np.zeros(units)
    for _ in range(draws):
        blocks.draw(rows, units, rng)
        sizes = np.bincount(blocks.labels[rows], minlength=units)
        assert sorted(sizes) == [2, 2, 3, 3]
        larger += sizes == 3
    # Each label holds a larger block half the time; binomial sd is ~32.
    assert np.all(np.abs(larger - draws / 2) < 5 * np.sqrt(draws / 4))


@pytest.mark.parametrize("equal_blocks", [False, True])
def test_hints_block_states_hold_exactly_their_rows(problem, target, equal_blocks):
    """
    A block state is never routed: its untouched leaves are placeholders read
    off the particle and the count table, its rewritten ones are split from
    their parent's. Whichever way a leaf got there, at every data node it has
    to hold exactly the rows the node's tree would route there from the node's
    block -- and the counts of those rows. And no placeholder may outlive its
    sweep: the particle handed back holds plain arrays.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.2, min_data=15,
                                 equal_blocks=equal_blocks)
    labels = proposal.blocks.labels
    node = proposal._node
    seen = {"nodes": 0, "unbuilt": 0}

    def checked(state, ctx, full, rng, lo, width, depth):
        fresh = idt.IncrementalTree.from_rows(problem, state.tree)
        for leaf in state._descendant_leaves(ctx.root):
            held = state.leaf_idx.held(leaf)
            if type(held) is BlockRows and held._rows is None:
                seen["unbuilt"] += 1
            want = fresh.leaf_idx[leaf]
            want = want[(labels[want] >= lo) & (labels[want] < lo + width)]
            assert np.array_equal(np.sort(state.leaf_idx[leaf]), np.sort(want))
            assert np.array_equal(state.counts[leaf],
                                  np.bincount(problem.y[want],
                                              minlength=problem.num_classes))
        seen["nodes"] += 1
        return node(state, ctx, full, rng, lo, width, depth)

    proposal._node = checked
    _, moves = walk(proposal, problem, seed=21, steps=300)
    assert len(moves) > 20
    assert seen["nodes"] > 200 and seen["unbuilt"] > 100
    for _, x in moves:
        assert type(x.leaf_idx) is dict


def test_hints_draws_a_new_split_every_sweep(problem, target):
    """
    HINTS splits the data afresh for every sweep. Every sweep that uses the
    hierarchy draws exactly one new split, of exactly the rows under that
    sweep's subtree root.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.2, min_data=15)
    draws = []
    draw = proposal.blocks.draw

    def watched(rows, n_units, rng):
        draws.append((np.sort(rows), n_units))
        return draw(rows, n_units, rng)

    proposal.blocks.draw = watched
    rng = RNG(5)
    x = idt.IncrementalTree.stump(problem)
    hierarchical = 0
    for _ in range(200):
        # the rows under each decision node of the tree this sweep starts from
        # (and every row, for the root of a stump, which is not a decision node)
        under = [np.sort(problem.all_rows())] + [
            np.sort(np.concatenate([x.leaf_idx[leaf]
                                    for leaf in x._descendant_leaves(node)]))
            for node in x.nodes]
        before = len(draws)
        x = proposal.sample(x, rng)
        if proposal._levels == 0:
            assert len(draws) == before
            continue
        hierarchical += 1
        assert len(draws) == before + 1
        rows, n_units = draws[-1]
        assert n_units == proposal._branching ** proposal._levels
        assert any(np.array_equal(rows, u) for u in under)
    assert hierarchical > 100


def test_hints_surrogate_at_full_data_is_the_target(problem, target):
    """
    The surrogate's departure from the target vanishes at f = 1: the
    tempering is a factor of f, and the counts are scaled by 1 / f. So the
    top of the hierarchy scores a tree exactly as the target does -- which is
    what makes the levels below it free to use whatever surrogate they like.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.25, min_data=20)
    _, moves = walk(idt.IncrementalTreeProposal(), problem, seed=8, steps=200)
    assert len(moves) > 20
    for _, x in moves:
        assert proposal._log_pi(x, 0, 1.0) == pytest.approx(target.eval(x), abs=1e-9)




def test_hints_drops_levels_rather_than_starve_them(problem, target):
    """
    A level that cannot be split into at least two blocks decides nothing and
    still costs a pass over its rows, so it is the levels that go, not the
    blocks.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.1, min_data=20, levels=3)
    assert proposal._shape(1) == (0, 0)
    assert proposal._shape(3)[0] == 1
    assert proposal._shape(8) == (3, 2)
    for total in (2, 5, 27, 64, 1000):
        levels, branching = proposal._shape(total)
        assert levels == 0 or branching ** levels <= total


@pytest.mark.parametrize("bad", [dict(branching=1), dict(branching=0),
                                 dict(levels=-1)])
def test_hints_refuses_settings_it_would_otherwise_quietly_change(target, bad):
    """branching=1 used to be clamped to 2, so a run reported a hierarchy it
    never built."""
    with pytest.raises(ValueError):
        idt.HINTSProposal(target, **bad)


def test_flat_hints_keeps_every_intermediate_state_admissible(problem, target):
    """
    A regression guard for the one bug in this domain that no identity test
    can see: a sweep that passes through a state with a starved leaf. From
    there a prune can merge the starved leaf away, while the grow that would
    undo it is barred -- a move with forward probability and no reverse path,
    which biases the sampled distribution by a couple of percent and nothing
    else.
    """
    proposal = idt.FlatHINTSProposal(target, ss_prop=0.2, min_data=15)
    proposal.validate_intermediate = True
    _, moves = walk(proposal, problem, seed=21, steps=400)
    assert len(moves) > 20



@pytest.mark.parametrize("levels, branching", [(1, None), (2, None), (3, 2)])
def test_hints_keeps_every_intermediate_state_admissible(problem, target,
                                                         levels, branching):
    """
    HINTS holds min_samples_leaf on the full rows at every level, through a
    full map it carries beside block states that only hold their block. After
    every applied primitive move the map has to name exactly the tree's leaves
    and none of them may be starved.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.05, min_data=5,
                                 levels=levels, branching=branching)
    proposal.validate_intermediate = True
    _, moves = walk(proposal, problem, seed=21, steps=300)
    assert len(moves) > 20
    if levels > 1:
        assert proposal.n_level_accepts > 0 and proposal.n_level_rejects > 0


def test_hints_full_rows_are_the_rows_the_tree_routes(problem, target):
    """
    The full map is never routed from scratch: it is the particle's leaves,
    brought forward by each move's own split. At every primitive block it has
    to hold, leaf for leaf, what routing every row through the block's current
    tree gives.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.2, min_data=15)
    primitive = proposal._primitive
    seen = {"blocks": 0}

    def checked(state, ctx, full, rng, f):
        fresh = idt.IncrementalTree.from_rows(problem, state.tree)
        assert set(full) == set(state._descendant_leaves(ctx.root))
        for leaf, rows in full.items():
            assert np.array_equal(np.sort(rows), np.sort(fresh.leaf_idx[leaf]))
        seen["blocks"] += 1
        return primitive(state, ctx, full, rng, f)

    proposal._primitive = checked
    _, moves = walk(proposal, problem, seed=17, steps=300)
    assert len(moves) > 20 and seen["blocks"] > 200


def test_hints_scales_each_block_by_its_actual_share(problem, target):
    """
    A block's f is the fraction of the subtree's rows that actually fell into
    it -- not the nominal 1 / U, which i.i.d. labels only hit on average.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.2, min_data=15)
    primitive = proposal._primitive
    seen = {"blocks": 0, "off_nominal": 0}

    def checked(state, ctx, full, rng, f):
        everything = sum(len(rows) for rows in full.values())
        mine = sum(len(state.leaf_idx[leaf])
                   for leaf in state._descendant_leaves(ctx.root))
        assert f == pytest.approx(mine / everything, abs=1e-12)
        if proposal._levels:
            nominal = 1 / proposal._branching ** proposal._levels
            seen["off_nominal"] += abs(f - nominal) > 1e-9
        seen["blocks"] += 1
        return primitive(state, ctx, full, rng, f)

    proposal._primitive = checked
    walk(proposal, problem, seed=4, steps=200)
    assert seen["blocks"] > 200 and seen["off_nominal"] > 100


def test_hints_rejects_moves_its_block_cannot_see(problem, target):
    """
    A primitive move whose block holds no row under the move node has nothing
    to be judged on, so it is screened out rather than decided by the prior.
    Blocks this small leave plenty of nodes with no rows in them.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.02, min_data=2, levels=2)
    proposal.record_moves = True
    walk(proposal, problem, seed=3, steps=300)
    log = proposal.move_log.arrays()
    blind = ((log['subset'] == 0) & (log['move'] != dg.MOVE_CODE["stay"])
             & (log['outcome'] != dg.BARRED))
    assert blind.sum() > 20
    assert np.all(log['outcome'][blind] == dg.SCREENED_OUT)


def test_hints_primitive_step_prices_the_surrogate_it_reports(problem, target):
    """
    A block accepts its move against one expression and hands its parent
    another -- a delta computed for the move, against a density computed over
    the whole subtree. The path correction is only a correction if those are
    the same function: the parent subtracts what the child charged, and a
    child that charged for something else leaves the difference in the
    acceptance ratio, where it biases the sweep silently.
    """
    proposal = idt.HINTSProposal(target, ss_prop=0.25, min_data=20)
    proposal.record_moves = True
    x = largest_tree_visited(problem, seed=5)
    leaves = x._descendant_leaves(0)
    rows = [x.leaf_idx[leaf] for leaf in leaves]
    full = {leaf: x.leaf_idx[leaf] for leaf in leaves}
    units = 60

    rng = RNG(4)
    checked = 0
    # Strict leaf sizes and blind blocks reject most tries.
    for _ in range(600):
        proposal.blocks.draw(np.concatenate(rows), units, rng)
        proposal.blocks.tabulate(rows, problem.num_classes)
        width = units // int(rng.nprng.integers(2, 6))
        lo = width * int(rng.nprng.integers(0, units // width))
        labels = proposal.blocks.labels
        pieces = []
        for r in rows:
            idx = r[(labels[r] >= lo) & (labels[r] < lo + width)]
            pieces.append((idx, np.bincount(problem.y[idx],
                                            minlength=problem.num_classes)))
        state = proposal._restricted(x, leaves, pieces)
        ctx = make_context(state, 0)
        f = proposal.blocks.share(lo, lo + width)
        assert f == pytest.approx(sum(len(p[0]) for p in pieces) / problem.n_rows)
        before = proposal._log_pi(state, 0, f)
        mark = len(proposal.move_log.arrays()['dsurr'])
        applied = proposal._primitive(state, ctx, full, rng, f)
        if not applied:
            continue
        after = proposal._log_pi(state, 0, f)
        log = proposal.move_log.arrays()
        charged = log['dsurr'][mark:][log['outcome'][mark:] == dg.APPLIED]
        assert len(applied) == 1 and len(charged) == 1
        assert after - before == pytest.approx(charged.sum(), abs=1e-8)
        checked += 1
    assert checked > 40


def test_hints_replays_a_sweep_onto_correctly_routed_rows(problem, target):
    """The particle a sweep hands back is the tree its moves build on every row."""
    proposal = idt.HINTSProposal(target, ss_prop=0.2, min_data=15)
    _, moves = walk(proposal, problem, seed=17, steps=400)
    assert len(moves) > 20
    for _, x in moves:
        assert type(x.leaf_idx) is dict
        fresh = idt.IncrementalTree.from_rows(problem, x.tree)
        assert set(fresh.leaf_idx) == set(x.leaf_idx)
        for leaf in fresh.leaf_idx:
            assert np.array_equal(np.sort(fresh.leaf_idx[leaf]),
                                  np.sort(x.leaf_idx[leaf]))
            assert np.array_equal(fresh.counts[leaf], x.counts[leaf])


# --------------------------------------------------------------------------- #
# tier 1: the samplers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["mh", "da", "flat_hints", "hints"])
def test_smc_returns_a_usable_sample(problem, target, name):
    """
    The SMC sampler must return a set of particles that are all admissible and have finite target mass.
    """
    proposals = {"mh": lambda: idt.IncrementalTreeProposal(),
                 "da": lambda: idt.DAProposal(target, ss_prop=0.25, min_data=20),
                 "flat_hints": lambda: idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20),
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
                                 proposal=proposal).sample(200, seed=0, verbose=False,
                                                           keep_samples=True)
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
            ).sample(iters, seed=seed, verbose=False, keep_samples=True)[burn:]
            sizes = idt.tree_sizes(chain)
            out.append([float(np.mean(sizes == k)) for k in range(3)])
        return np.array(out)

    reference = pmf(lambda: idt.IncrementalTreeProposal())
    for name, make in [("DA", lambda: idt.DAProposal(target, ss_prop=0.25, min_data=20)),
                       ("FlatHINTS", lambda: idt.FlatHINTSProposal(target, ss_prop=0.25,
                                                                   min_data=20)),
                       # 300 rows into 10 blocks: two levels of 3, so the
                       # intermediate accept runs with min_samples_leaf binding
                       # -- held on the full rows at every level.
                       ("HINTS", lambda: idt.HINTSProposal(target, ss_prop=0.1,
                                                           min_data=15))]:
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


@pytest.mark.parametrize("name", ["MH", "DA", "FlatHINTS", "HINTS"])
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
        "FlatHINTS": lambda: idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20),
        # levels are reduced to what 60 rows can feed, so this runs at L=1;
        # the two-level hierarchy is graded against the same reference in
        # test_hints_hierarchy_matches_the_exact_posterior.
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
        ).sample(40_000, seed=seed, verbose=False, keep_samples=True)[5_000:]
        estimates.append(float(np.mean(idt.tree_sizes(chain) == 0)))

    estimates = np.array(estimates)
    se = estimates.std(ddof=1) / np.sqrt(len(estimates))
    diff = estimates.mean() - exact
    assert abs(diff) < 4.0 * se + 0.003, (
        f"{name} disagrees with the exact posterior on P(0 nodes): "
        f"{estimates.mean():.4f} vs {exact:.4f} "
        f"({diff / max(se, 1e-12):+.1f} sigma)")


@pytest.mark.parametrize("equal_blocks", [False, True])
def test_hints_hierarchy_matches_the_exact_posterior(equal_blocks):
    """
    The same exact reference as test_kernels_match_the_exact_posterior, against
    a hierarchy that really is one.

    60 rows only feed one level at the settings used there, so that run grades
    HINTS as a flat sweep and says nothing about the part that is new: the
    intermediate accept step, and the path correction that has to cancel it. A
    level whose surrogate and reported density disagreed, or one that kept a
    move its parent had rejected, shows up here and nowhere else.
    """
    problem = make_problem(n=60, d=2, seed=0, lam=1.0, min_samples_leaf=0,
                           max_tree_size=1)
    problem.y = np.random.default_rng(0).integers(
        0, 2, size=problem.n_rows).astype(np.int64)
    problem._root_counts = None

    target = idt.IncrementalTreeTarget(problem)
    init = idt.IncrementalTreeInitialProposal(problem)
    exact = exact_p_stump(problem, target)
    assert idt.HINTSProposal(target, ss_prop=0.1, min_data=4)._shape(10) == (2, 3)

    estimates = []
    for seed in range(8):
        chain = DiscreteVariableMCMC(
            idt.IncrementalTree, target, init,
            proposal=idt.HINTSProposal(target, ss_prop=0.1, min_data=4, levels=2,
                                       equal_blocks=equal_blocks)
        ).sample(40_000, seed=seed, verbose=False, keep_samples=True)[5_000:]
        estimates.append(float(np.mean(idt.tree_sizes(chain) == 0)))

    estimates = np.array(estimates)
    se = estimates.std(ddof=1) / np.sqrt(len(estimates))
    diff = estimates.mean() - exact
    assert abs(diff) < 4.0 * se + 0.003, (
        f"the HINTS hierarchy disagrees with the exact posterior on "
        f"P(0 nodes): {estimates.mean():.4f} vs {exact:.4f} "
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
        ).sample(40_000, seed=seed, verbose=False, keep_samples=True)[5_000:]
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


# --------------------------------------------------------------------------- #
# instrumentation: the move log, the SMC diagnostics and the metrics
# --------------------------------------------------------------------------- #

def _chain(problem, target, proposal, iters=300, seed=1):
    mcmc = DiscreteVariableMCMC(idt.IncrementalTree, target,
                                idt.IncrementalTreeInitialProposal(problem),
                                proposal=proposal)
    trace = []
    mcmc.sample(iters, seed=seed, verbose=False,
                callback=lambda i, c, a: trace.append((len(c.tree), a)))
    return mcmc, trace


def _make_proposal(name, target):
    if name == "MH":
        return idt.IncrementalTreeProposal()
    if name == "DA":
        return idt.DAProposal(target, ss_prop=0.25, min_data=20)
    if name == "HINTS":
        return idt.HINTSProposal(target, ss_prop=0.25, min_data=20)
    return idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20)


@pytest.mark.parametrize("name", ["MH", "DA", "FlatHINTS", "HINTS"])
def test_recording_moves_does_not_change_the_chain(problem, target, name):
    """
    The whole point of the log is to compare samplers, so it must not perturb
    the one it is watching. It draws no random numbers, so the chain a run
    produces has to be identical bit for bit with recording on and off.
    """
    off = _chain(problem, target, _make_proposal(name, target))[1]
    proposal = _make_proposal(name, target)
    proposal.record_moves = True
    assert _chain(problem, target, proposal)[1] == off


@pytest.mark.parametrize("name", ["MH", "DA", "FlatHINTS", "HINTS"])
def test_every_call_closes_out_exactly_one_call_row(problem, target, name):
    """
    A call that recorded no outcome would silently drop moves from the counts;
    one that recorded two would double-count them. Every exit has to close the
    call exactly once.
    """
    proposal = _make_proposal(name, target)
    proposal.record_moves = True
    _chain(problem, target, proposal, iters=400)
    log = proposal.move_log.arrays()
    assert len(log['call_id']) == proposal.n_calls
    assert np.array_equal(log['call_id'], np.arange(proposal.n_calls))
    # No move may be attributed to a call that never happened.
    assert log['call'].max() < proposal.n_calls


@pytest.mark.parametrize("name", ["MH", "DA"])
def test_one_move_per_call_and_its_fate_is_the_calls(problem, target, name):
    """
    MH and DA consider exactly one move per iteration, so the move's fate and
    the call's are the same event seen twice, and the counters have to total
    them. The comparison against HINTS rests on that: HINTS spends several
    moves where these spend one.
    """
    proposal = _make_proposal(name, target)
    proposal.record_moves = True
    _chain(problem, target, proposal, iters=400)
    log = proposal.move_log.arrays()
    counters = proposal.counters()

    assert len(log['move']) == proposal.n_calls
    assert np.array_equal(log['outcome'], log['call_outcome'])

    table = proposal.move_log.counts()
    # A "stay" is the absence of a move, so it can have no other fate.
    assert table[dg.MOVE_CODE['stay']].sum() == table[dg.MOVE_CODE['stay'], dg.STAY]
    assert table[:, dg.STAY].sum() == counters['stays'] - counters['barred'] \
        - counters['screened_out'] - counters['inadmissible']
    assert table[:, dg.BARRED].sum() == counters['barred']
    assert table[:, dg.SCREENED_OUT].sum() == counters['screened_out']
    assert table[:, dg.INADMISSIBLE].sum() == counters['inadmissible']
    # n_moved counts the calls that produced a new state, which is exactly the
    # moves handed on to the outer step. n_inner_moves is a looser count: it
    # rises before _finish, which can still bar the move afterwards.
    assert table[:, dg.PROPOSED].sum() == counters['moved']
    assert table[:, dg.APPLIED].sum() == 0
    assert counters['inner_moves'] >= counters['moved']


def test_flat_hints_move_counts_total_the_sweep_counters(problem, target):
    proposal = idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20)
    proposal.record_moves = True
    _chain(problem, target, proposal, iters=400)
    log = proposal.move_log.arrays()
    counters = proposal.counters()
    table = proposal.move_log.counts()

    # One row per block, and every block emits exactly one.
    assert len(log['move']) == counters['blocks']
    assert len(log['move']) > proposal.n_calls
    assert table[dg.MOVE_CODE['stay']].sum() == table[dg.MOVE_CODE['stay'], dg.STAY]
    assert table[:, dg.SCREENED_OUT].sum() == counters['screened_out']
    assert table[:, dg.INADMISSIBLE].sum() == counters['inadmissible']
    assert table[:, dg.APPLIED].sum() == counters['inner_moves']
    assert table[:, dg.PROPOSED].sum() == 0, "a block move is applied, not proposed"

    # A sweep can be barred at two levels: an individual block's move, and the
    # root correction on the sweep as a whole. The counter totals both.
    call_barred = int((log['call_outcome'] == dg.BARRED).sum())
    assert table[:, dg.BARRED].sum() + call_barred == counters['barred']
    assert int((log['call_outcome'] == dg.PROPOSED).sum()) == counters['moved']


def test_pruning_the_subtree_root_is_barred_at_the_move(problem):
    """
    A prune of the subtree root is out of the root-draw's support, so
    evaluate_subtree_move bars it -- unless it is the last split, whose prune
    leaves a stump. Every proposal reads the bar off the same call, so a sweep
    loses that one move, as MH and DA do, rather than the whole sweep.
    """
    rng = RNG(3)
    x = largest_tree_visited(problem, seed=3)
    assert len(x.tree) > 1
    for root in x.terminal_nodes:
        ctx = make_context(x, root)
        pc, _ = evaluate_subtree_move(x, ctx, "prune", root, None, rng)
        assert pc <= idt.BARRED

    one = idt.IncrementalTree.stump(problem)
    one.grow(0, 0, 0.0)
    pc, _ = evaluate_subtree_move(one, make_context(one, 0), "prune", 0, None, rng)
    assert pc > idt.BARRED


@pytest.mark.parametrize("name", ["FlatHINTS", "HINTS"])
def test_sweeps_are_never_barred_by_the_root_draw(problem, target, name):
    """
    With root prunes barred move by move, no sweep can end without its subtree
    root, so the sweep-level root check never throws a sweep's moves away.
    """
    proposal = _make_proposal(name, target)
    proposal.record_moves = True
    _chain(problem, target, proposal, iters=400)
    log = proposal.move_log.arrays()
    assert int((log['call_outcome'] == dg.BARRED).sum()) == 0
    assert int((log['outcome'] == dg.BARRED).sum()) == proposal.counters()['barred']


def test_screened_moves_carry_the_subsample_they_were_judged_on(problem, target):
    """
    The subsample size and surrogate ratio are the columns that say what a
    delayed-acceptance screen actually cost and decided. A move that reached the
    surrogate carries both; one that never got there carries neither.
    """
    proposal = idt.DAProposal(target, ss_prop=0.25, min_data=20)
    proposal.record_moves = True
    _chain(problem, target, proposal, iters=400)
    log = proposal.move_log.arrays()

    # dsurr, not subset, is what marks a screened move: a block can legitimately
    # hold none of the node's rows, and that empty screen is still a screen.
    screened = ~np.isnan(log['dsurr'])
    assert screened.any()
    assert (log['subset'][~screened] == 0).all()
    # Only a real move is ever screened, and only a screened one is screened out.
    assert not (log['outcome'][screened] == dg.STAY).any()
    assert screened[log['outcome'] == dg.SCREENED_OUT].all()
    # A subsample cannot hold more rows than the node it was drawn from.
    assert (log['subset'][screened] <= log['rows'][screened]).all()

    mh = idt.IncrementalTreeProposal()
    mh.record_moves = True
    _chain(problem, target, mh, iters=200)
    # MH screens nothing, so it has no subsample to report.
    assert (mh.move_log.arrays()['subset'] == 0).all()


def test_flat_hints_records_one_row_per_block_of_the_sweep(problem, target):
    """
    The per-subsample level the sweep works at: each block draws its own move
    and screens it on its own rows, and each is a row of the log tagged with
    the call it belongs to.
    """
    proposal = idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20)
    proposal.record_moves = True
    _chain(problem, target, proposal, iters=300)
    log = proposal.move_log.arrays()

    per_call = np.bincount(log['call'], minlength=proposal.n_calls)
    assert (per_call >= 1).all(), "every sweep runs at least one block"
    assert per_call.sum() == proposal.counters()['blocks']
    # A sweep only reaches the outer step if it applied something, and a sweep
    # that applied nothing cannot have got there.
    applied = np.bincount(log['call'][log['outcome'] == dg.APPLIED],
                          minlength=proposal.n_calls)
    proposed = log['call_outcome'] == dg.PROPOSED
    assert (applied[proposed] > 0).all()
    assert not proposed[applied == 0].any()


def test_smc_records_ess_and_returns_usable_weights(problem, target):
    proposal = idt.FlatHINTSProposal(target, ss_prop=0.25, min_data=20)
    smc = DiscreteVariableSMC(idt.IncrementalTree, target,
                              idt.IncrementalTreeInitialProposal(problem),
                              proposal=proposal, Lkernel=proposal.lkernel())
    seen = []
    particles = smc.sample(8, 32, seed=1, verbose=False,
                           callback=lambda t, p, lw, n, r: seen.append((t, n, r)))

    assert [s[0] for s in seen] == list(range(8))
    # One per step, plus the final weights the loop never normalises itself.
    assert len(smc.ess_history) == 9
    assert len(smc.resampled_history) == 8
    assert [s[1] for s in seen] == smc.ess_history[:8]
    assert [s[2] for s in seen] == smc.resampled_history
    assert all(1.0 <= e <= 32.0 + 1e-9 for e in smc.ess_history)

    weights = np.exp(smc.logWeights)
    assert weights.sum() == pytest.approx(1.0)
    assert len(weights) == len(particles)
    # Weighted and unweighted ensembles are different estimators; both must at
    # least be proper distributions over the classes.
    for w in (weights, None):
        probs = idt.ensemble_predict_proba(particles, problem.X, w)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_classification_metrics_agree_with_sklearn(problem, target):
    from sklearn import metrics as skm

    x, _ = walk(idt.IncrementalTreeProposal(), problem, seed=4, steps=300)
    proba = idt.predict_proba(x, problem.X)
    y = problem.y
    got = idt.classification_metrics(y, proba, problem.num_classes, prefix="t_")
    pred = np.argmax(proba, axis=1)

    assert np.array_equal(got['t_confusion'], skm.confusion_matrix(y, pred))
    assert got['t_accuracy'] == pytest.approx(skm.accuracy_score(y, pred))
    assert got['t_balanced_accuracy'] == pytest.approx(
        skm.balanced_accuracy_score(y, pred))
    assert got['t_macro_f1'] == pytest.approx(
        skm.f1_score(y, pred, average='macro', zero_division=0))
    assert got['t_macro_precision'] == pytest.approx(
        skm.precision_score(y, pred, average='macro', zero_division=0))
    assert got['t_macro_recall'] == pytest.approx(
        skm.recall_score(y, pred, average='macro', zero_division=0))
    assert got['t_log_loss'] == pytest.approx(
        skm.log_loss(y, proba, labels=list(range(problem.num_classes))))
    # Multiclass Brier, summed over classes, which sklearn has no direct entry
    # point for in this form.
    onehot = np.eye(problem.num_classes)[y]
    assert got['t_brier'] == pytest.approx(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def test_state_metrics_agree_with_row_by_row(problem):
    """Scoring per-leaf tallies gives the same numbers as scoring each row."""
    x, _ = walk(idt.IncrementalTreeProposal(), problem, seed=4, steps=300)
    rs = np.random.default_rng(7)
    # Rows the tree was not fitted to, labelled independently of it.
    X = problem.X + 0.3 * rs.normal(size=problem.X.shape)
    y = rs.integers(problem.num_classes, size=len(X))
    got = idt.state_metrics(x, X, y, prefix="t_")
    want = idt.classification_metrics(y, idt.predict_proba(x, X),
                                      problem.num_classes, prefix="t_")
    for key, value in want.items():
        assert np.allclose(got[key], value), key


def test_evaluate_is_the_weighted_mean_of_each_trees_metrics(problem):
    trees = [walk(idt.IncrementalTreeProposal(), problem, seed=s, steps=200)[0]
             for s in range(3)]
    w = np.array([0.2, 0.5, 0.3])
    got = idt.evaluate(trees, problem.X, problem.y, weights=w)
    each = [idt.state_metrics(t, problem.X, problem.y) for t in trees]
    for key, value in got.items():
        assert np.allclose(value, sum(wi * m[key] for wi, m in zip(w, each))), key


def test_metrics_survive_a_class_that_is_never_predicted():
    """
    A short chain can sit on a stump, which predicts one class for everything.
    Precision for the classes it never names is 0/0, and the run has to carry on
    with a number rather than a nan.
    """
    problem = make_problem(n=120, d=3, seed=2)
    stump = idt.IncrementalTree.stump(problem)
    got = idt.classification_metrics(problem.y, idt.predict_proba(stump, problem.X),
                                     problem.num_classes)
    assert np.isfinite(got['macro_precision'])
    assert np.isfinite(got['macro_f1'])
    assert got['confusion'].sum() == problem.n_rows
