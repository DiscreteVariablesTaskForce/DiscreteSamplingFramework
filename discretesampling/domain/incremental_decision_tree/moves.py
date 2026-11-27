from collections import deque
from math import exp, log

import numpy as np
from scipy.special import gammaln

from discretesampling.domain.incremental_decision_tree.problem import large_neg

MOVE_PROBS = {"grow": 0.35, "prune": 0.30, "change": 0.35}
MOVE_PROBS_AT_CAP = {"grow": 0.0, "prune": 0.4, "change": 0.6}

_MOVE_NAMES = tuple(MOVE_PROBS)
_MOVE_CUM = np.cumsum([MOVE_PROBS[m] for m in _MOVE_NAMES])
_MOVE_CUM_AT_CAP = np.cumsum([MOVE_PROBS_AT_CAP[m] for m in _MOVE_NAMES])


class SubtreeContext:
    """
    The nodes, terminal nodes and leaves under one root, plus that subtree's
    slice of leaf_idx.
    """
    __slots__ = ('root', 'nodes', 'terminal_nodes', 'leaves', 'leaf_idx')

    def __init__(self, root, nodes, terminal_nodes, leaves, leaf_idx):
        self.root = root
        self.nodes = nodes
        self.terminal_nodes = terminal_nodes
        self.leaves = leaves
        self.leaf_idx = leaf_idx

    def __repr__(self):
        return (f"<SubtreeContext root={self.root} {len(self.nodes)} nodes, "
                f"{len(self.leaves)} leaves>")


def _leaf_idx_of(state, leaves):
    """
    state.leaf_idx restricted to `leaves`. A HINTS block state's mapping
    builds a leaf's rows only when they are read, and sub() keeps it that way;
    reading every leaf here would build them all.
    """
    leaf_idx = state.leaf_idx
    sub = getattr(leaf_idx, 'sub', None)
    if sub is not None:
        return sub(leaves)
    return {leaf: leaf_idx[leaf] for leaf in leaves if leaf in leaf_idx}


def make_context(state, root):
    """
    One walk of the subtree under `root`, collecting everything a move needs.
    """
    if not state.tree:
        leaves = [root]
        return SubtreeContext(root, [], [], leaves, _leaf_idx_of(state, leaves))
    m = state.nodes
    stack, nodes, terminal_nodes, leaves = [root], [], [], []
    while stack:
        n = stack.pop()
        if n in m:
            nodes.append(n)
            L, R = int(m[n][1]), int(m[n][2])
            if L not in m and R not in m:
                terminal_nodes.append(n)
            stack.extend([L, R])
        else:
            leaves.append(n)
    return SubtreeContext(root, nodes, terminal_nodes, leaves,
                          _leaf_idx_of(state, leaves))


def draw_subtree(state, rng):
    """
    This iteration's subtree root, and its context.
    The root is drawn uniformly from the internal nodes of the tree, or from
    the singleton set {0} if the tree is empty."""
    pool = state.list_nodes or [0]
    root = pool[int(rng.nprng.integers(0, len(pool)))]
    return root, make_context(state, root)


# ------------------- reading rows out of a context ------------------- #

def subtree_node_data(state, ctx, node):
    """
    Row indices under `node`, read out of the context rather than the tree.
    """
    if node is None:
        return np.array([], dtype=int)
    if node in ctx.leaf_idx:
        return ctx.leaf_idx[node]
    arrs = [ctx.leaf_idx[leaf] for leaf in state._descendant_leaves(node)
            if leaf in ctx.leaf_idx]
    return np.concatenate(arrs) if arrs else np.array([], dtype=int)


def subtree_leaves(state, ctx, node):
    """
    The leaves under `node`, in the order subtree_node_data concatenates their
    rows.
    """
    if node in ctx.leaf_idx:
        return [node]
    return [leaf for leaf in state._descendant_leaves(node) if leaf in ctx.leaf_idx]


def subtree_node_data_len(state, ctx, node):
    """
    How many rows are under `node`, without materialising them.
    """
    if node is None:
        return 0
    if node in ctx.leaf_idx:
        return len(ctx.leaf_idx[node])
    return sum(len(ctx.leaf_idx[leaf])
               for leaf in state._descendant_leaves(node)
               if leaf in ctx.leaf_idx)


def subtree_size(ctx):
    """
    Rows under ctx.root

    subtree_node_data_len re-derives the subtree's leaves with a tree walk,
    which is right for an arbitrary node and pure waste for the context's own
    root: make_context already resolved exactly that set into ctx.leaf_idx.
    """
    return sum(len(a) for a in ctx.leaf_idx.values())


# ------------------- move selection ------------------- #

def move_probs(problem, n_tree_nodes):
    """
    The move probabilities, depending on tree size.
    At max_tree_size, grow is barred.
    """
    return MOVE_PROBS_AT_CAP if n_tree_nodes >= problem.max_tree_size else MOVE_PROBS


def select_move(state, ctx, rng):
    """
    Draw a move type and a valid node from ctx for it,
    if possible, else "stay" and None.
    Also "stay" if the node drawn produces child leaves
    that violate min_samples_leaf.
    """
    problem = state.problem
    at_cap = move_probs(problem, len(state.tree)) is MOVE_PROBS_AT_CAP
    cum = _MOVE_CUM_AT_CAP if at_cap else _MOVE_CUM
    move = _MOVE_NAMES[int(np.searchsorted(cum, rng.random(), side='right'))]

    if move == "grow":
        pool = ctx.leaves
    elif move == "change":
        pool = ctx.nodes
    else:
        pool = ctx.terminal_nodes
    if not pool:
        return "stay", None

    node = pool[int(rng.nprng.integers(0, len(pool)))]
    if (move in ("grow", "change")
            and subtree_node_data_len(state, ctx, node) < 2 * problem.min_samples_leaf):
        return "stay", None
    return move, node


def subtree_admissible(state, subtree_root):
    """
    Check whether every leaf under `subtree_root` still meets min_samples_leaf.
    """
    min_leaf = state.problem.min_samples_leaf
    return all(len(state.leaf_idx[leaf]) >= min_leaf
               for leaf in state._descendant_leaves(subtree_root))


def change_partition(state, node, feat, thr, idx):
    """
    The rows `idx` under `node` re-routed as a change of its split to
    (feat, thr) would route them: {leaf: rows}, in the order IncBDTree.change
    builds its leaves -- or None as soon as a leaf would hold fewer than
    min_samples_leaf rows. Routing them is the whole cost of applying the
    change, so a change that passes is applied from this instead of routed
    a second time.
    """
    m = state.nodes
    X = state.problem.X
    reads = state.problem.reads
    min_leaf = state.problem.min_samples_leaf
    partition = {}
    queue = deque([(node, idx, True)])
    while queue:
        curr, curr_idx, is_changed = queue.popleft()
        row = m.get(curr)
        if row is None:
            if len(curr_idx) < min_leaf:
                return None
            partition[curr] = curr_idx
            continue
        curr_feat = int(feat) if is_changed else int(row[3])
        curr_thr = float(thr) if is_changed else float(row[4])
        reads.n += len(curr_idx)
        left = X[curr_idx, curr_feat] < curr_thr
        queue.append((int(row[1]), curr_idx[left], False))
        queue.append((int(row[2]), curr_idx[~left], False))
    return partition


def change_admissible(state, node, feat, thr, idx):
    """
    Check whether a change move leaves every leaf under `node` still meets
    min_samples_leaf. This is the admissibility check for a change move only,
    because grow and prune are already screened by select_move.
    """
    return change_partition(state, node, feat, thr, idx) is not None


def valid_split(problem, go_left):
    """
    Check whether a split sending the rows marked in `go_left` left produces
    child leaves that meet min_samples_leaf.
    """
    left = np.count_nonzero(go_left)
    return left >= problem.min_samples_leaf and (len(go_left) - left) >= problem.min_samples_leaf


_SPLIT_PROBE = 256


def check_grow_split(problem, idx, feat, thr):
    """
    (valid, go_left) for splitting the rows `idx` on (feat, thr): whether
    both children meet min_samples_leaf, and the split's mask when the full
    rows had to be split to tell. The first rows are tried alone first --
    once they already put enough rows on each side the split is valid,
    whatever the rest do, and the mask is left for grow to compute only if
    the move is actually applied.
    """
    head = problem.X[idx[:_SPLIT_PROBE], feat] < thr
    problem.reads.n += len(head)
    if len(idx) > _SPLIT_PROBE and valid_split(problem, head):
        return True, None
    if len(idx) <= _SPLIT_PROBE:
        go_left = head
    else:
        go_left = problem.X[idx, feat] < thr
        problem.reads.n += len(idx)
    return valid_split(problem, go_left), go_left


def evaluate_subtree_move(state, ctx, move, node, idx, rng):
    """
    Evaluate a move, returning the log proposal correction:
    log q(x | x') - log q(x' | x),
    and the information needed to apply the move.
    """
    problem = state.problem
    stump = len(state.tree) == 0
    num_terminal_nodes = len(ctx.terminal_nodes)
    num_leaves = len(ctx.leaves)
    feat = thr = None

    if move == "stay":
        return 0.0, {'node': node, 'feat': None, 'thr': None, 'stump': stump}

    # Forward and backward move-type probabilities are read off the tree size
    # on their own side of the move; they differ only near max_tree_size.
    mp_f = move_probs(problem, len(state.tree))
    mp_b = move_probs(problem, len(state.tree) + {"grow": 1, "prune": -1}.get(move, 0))

    informed = problem.split_proposal == "informed"
    xbart = problem.split_proposal == "xbart"
    # Lakshminarayanan et al.'s optimal proposal on a random subset of the rows.
    marginal = problem.split_proposal == "marginal"
    # Chipman et al.'s node values for the threshold; the feature stays uniform.
    node_thr = problem.threshold_proposal == "node" and not (informed or xbart or marginal)

    if move == "grow":
        if xbart:
            cands = _xbart_splits(problem, ctx.leaf_idx[node])
            feat, thr, lq_split = _draw_xbart(problem, cands, rng)
        elif marginal:
            # S from the leaf's rows as this sampler sees them, as for "informed".
            cands = _marginal_splits(problem, informed_subset(problem, ctx.leaf_idx[node], rng))
            feat, thr, lq_split = _draw_xbart(problem, cands, rng)
        elif informed:
            # S comes from the leaf's rows as this sampler sees them (a HINTS
            # unit vertex: its block; MH and DA: all of them), the same rows the
            # reverse prune will draw its S from.
            log_w, intervals = _best_splits(problem, informed_subset(problem, ctx.leaf_idx[node], rng))
            feat, thr, lq_split = _draw_informed(problem, log_w, intervals, rng)
        elif node_thr:
            feat = problem.random_feature(rng)
            thr, lq_thr = _node_threshold(problem, ctx.leaf_idx[node], feat, rng)
            lq_split = lq_thr + problem.lp_feats
        else:
            feat = problem.random_feature(rng)
            thr, lp_thr = problem.random_threshold(feat, rng)
            lq_split = lp_thr + problem.lp_feats
        # The mask, when computed, is carried on the move so applying it does
        # not split the same rows again.
        valid, go_left = check_grow_split(problem, idx, feat, thr)
        if not valid:
            return large_neg, {'node': node, 'feat': feat, 'thr': thr, 'stump': stump}

        q_f = mp_f[move] / (1.0 if stump else float(num_leaves))

        # The reverse prune draws from the terminal nodes of the PROPOSED tree.
        # Splitting `node` makes it terminal, but also stops its parent being
        # terminal whenever the sibling is a leaf, so the count does not simply
        # rise by one. Getting this wrong under-weights grow by |T|/(|T|+1).
        denom_b = num_terminal_nodes + 1.0
        if node != ctx.root:
            parent = state._parent_of(node)
            if parent is not None:
                L, R = state._child_nodes(parent)
                if (R if L == node else L) not in state.nodes:
                    denom_b -= 1.0
        q_b = mp_b["prune"] / denom_b
        if q_f <= 0.0 or q_b <= 0.0:
            return large_neg, {'node': node, 'feat': feat, 'thr': thr, 'stump': stump}
        prop_correction = log(q_b) - log(q_f) - lq_split

    elif move == "prune":
        if node == ctx.root and len(state.tree) > 1:
            # Pruning the subtree root leaves it undrawable on the way back, so
            # the root-draw term bars it (_root_correction). Barred here rather
            # than there, so that a HINTS sweep loses this one move, as MH and
            # DA do, instead of finding out at the end of the sweep and
            # discarding every move it made. The last split is the exception:
            # its prune leaves a stump, whose root is drawn from {0}.
            return large_neg, {'node': node, 'feat': None, 'thr': None, 'stump': stump}
        old_feat = int(state.nodes[node][3])
        # The reverse grow would have to re-draw this threshold, so this is the
        # PROPOSAL density, not the prior -- the two differ once
        # threshold_proposal is not "uniform".
        lp_old_thr = problem.lp_thr_proposal(old_feat, float(state.nodes[node][4]))

        q_f = mp_f[move] / (1.0 if stump else float(num_terminal_nodes))
        q_b = mp_b["grow"] / max(1.0, num_leaves - 1.0)
        if q_f <= 0.0 or q_b <= 0.0:
            # Reverse grow is barred at the cap: the return path has zero
            # probability, so the prune has to be rejected outright.
            return large_neg, {'node': node, 'feat': None, 'thr': None, 'stump': stump}
        if xbart:
            lq_old = _lq_xbart(problem, _xbart_splits(problem, subtree_node_data(state, ctx, node)),
                               old_feat, float(state.nodes[node][4]))
        elif marginal:
            # The reverse grow's density, from an S drawn from the rows the grow drew from.
            cands = _marginal_splits(
                problem, informed_subset(problem, subtree_node_data(state, ctx, node), rng))
            lq_old = _lq_xbart(problem, cands, old_feat, float(state.nodes[node][4]))
        elif informed:
            # The reverse grow's density of re-drawing this split, from an S
            # drawn from the node's rows: the rows the grow drew from.
            log_w, intervals = _best_splits(
                problem, informed_subset(problem, subtree_node_data(state, ctx, node), rng))
            old_thr = float(state.nodes[node][4])
            lq_old = log_w[old_feat] + _lq_thr(problem, old_feat, old_thr, intervals[old_feat])
        elif node_thr:
            lq_old = problem.lp_feats + _lq_node_threshold(
                problem, _node_values(problem, subtree_node_data(state, ctx, node), old_feat),
                old_feat, float(state.nodes[node][4]))
        else:
            lq_old = lp_old_thr + problem.lp_feats
        prop_correction = log(q_b) + lq_old - log(q_f)

    elif marginal:  # change: one S for both directions (the node keeps its rows)
        cands = _marginal_splits(
            problem, informed_subset(problem, subtree_node_data(state, ctx, node), rng))
        feat, thr, lq_new = _draw_xbart(problem, cands, rng)
        old_feat = int(state.nodes[node][3])
        prop_correction = (_lq_xbart(problem, cands, old_feat, float(state.nodes[node][4]))
                           - lq_new)

    elif xbart:  # change, XBART-style: the node's rows give both directions
        cands = _xbart_splits(problem, subtree_node_data(state, ctx, node))
        feat, thr, lq_new = _draw_xbart(problem, cands, rng)
        old_feat = int(state.nodes[node][3])
        prop_correction = (_lq_xbart(problem, cands, old_feat, float(state.nodes[node][4]))
                           - lq_new)

    elif node_thr:  # change with node values: the node keeps its rows
        rows = subtree_node_data(state, ctx, node)
        feat = problem.random_feature(rng)
        thr, lq_new = _node_threshold(problem, rows, feat, rng)
        old_feat = int(state.nodes[node][3])
        prop_correction = (_lq_node_threshold(problem, _node_values(problem, rows, old_feat), old_feat,
                                              float(state.nodes[node][4]))
                           - lq_new)

    elif informed:  # change, informed: one S for both directions
        log_w, intervals = _best_splits(
            problem, informed_subset(problem, subtree_node_data(state, ctx, node), rng))
        feat, thr, lq_new = _draw_informed(problem, log_w, intervals, rng)
        old_feat = int(state.nodes[node][3])
        old_thr = float(state.nodes[node][4])
        prop_correction = (log_w[old_feat] + _lq_thr(problem, old_feat, old_thr, intervals[old_feat])
                           - lq_new)

    else:  # change
        feat = problem.random_feature(rng)
        thr, lp_thr = problem.random_threshold(feat, rng)
        # No min_samples_leaf test here. A change re-routes the whole subtree,
        # so the split it lands on says nothing about the leaves further down,
        # and screening the top split alone rejected forward moves without
        # rejecting the matching reverse. The sound test is on the resulting
        # tree, which is what the admissibility check after the move makes.
        old_feat = int(state.nodes[node][3])
        # Reverse density of re-drawing the threshold this change overwrites:
        # again the PROPOSAL density, not the prior.
        prop_correction = (problem.lp_thr_proposal(old_feat, float(state.nodes[node][4]))
                           - lp_thr)

    prop_move = {'node': node, 'feat': feat, 'thr': thr, 'stump': stump}
    if move == "grow" and go_left is not None:
        prop_move['go_left'] = go_left
    return prop_correction, prop_move


def apply_subtree_proposal(state, ctx, move, prop_move):
    """
    Apply a move to the state, updating the context in place.
    Returns the new state, which is the same object as the input.
    """
    if move == "stay":
        return state
    node = prop_move['node']

    if move == "grow":
        state.grow(node, prop_move['feat'], prop_move['thr'], prop_move.get('go_left'))
        L, R = state._child_nodes(node)
        ctx.nodes.append(node)
        ctx.terminal_nodes.append(node)
        ctx.leaves.remove(node)
        ctx.leaves.extend([L, R])
        del ctx.leaf_idx[node]
        ctx.leaf_idx[L] = state.leaf_idx[L]
        ctx.leaf_idx[R] = state.leaf_idx[R]
        # The parent stops being terminal once `node` is internal.
        parent = state._parent_of(node)
        if node != ctx.root and parent in ctx.terminal_nodes:
            ctx.terminal_nodes.remove(parent)

    elif move == "prune":
        L, R = state._child_nodes(node)
        state.prune(node)
        ctx.nodes.remove(node)
        ctx.terminal_nodes.remove(node)
        ctx.leaves.remove(L)
        ctx.leaves.remove(R)
        ctx.leaves.append(node)
        ctx.leaf_idx.pop(L, None)
        ctx.leaf_idx.pop(R, None)
        ctx.leaf_idx[node] = state.leaf_idx[node]
        # The parent becomes terminal if its other child is a leaf too.
        parent = state._parent_of(node)
        if node != ctx.root and parent is not None:
            pL, pR = state._child_nodes(parent)
            if pL not in state.nodes and pR not in state.nodes:
                ctx.terminal_nodes.append(parent)

    else:  # change: only the routing below `node` moves
        state.change(node, prop_move['feat'], prop_move['thr'], prop_move.get('partition'))
        for leaf in state._descendant_leaves(node):
            ctx.leaf_idx[leaf] = state.leaf_idx[leaf]

    return state


# ------------------- the subtree-root auxiliary variable ------------------- #

def _root_correction(old_len, new_len, root_survives, subtree_root):
    """
    The auxiliary-variable correction for a subtree move:
    log p(root | x') - log p(root | x)
    The probability of the root draw is uniform over the internal nodes, so the
    correction is the log of the ratio of the number of internal nodes on each
    side of the move. If the subtree root is pruned, it is no longer drawable
    and the forward probability is zero, so the move is barred. If the tree is
    empty, the root is drawn from a singleton set and the correction is zero.
    """
    if new_len:
        if not root_survives:
            return large_neg
        denom_new = new_len
    elif subtree_root != 0:
        return large_neg
    else:
        denom_new = 1
    return log(max(old_len, 1)) - log(denom_new)


def root_correction(state, state_new, subtree_root):
    """The root-draw term, read off a move that has already been applied."""
    return _root_correction(len(state.tree), len(state_new.tree),
                            subtree_root in state_new.nodes, subtree_root)


def proposal_root_correction(state, move, node, subtree_root):
    """
    The root-draw term, read off a move that has not yet been applied.
    This is the same as root_correction, but the new tree size is inferred from
    the move type rather than the new state.
    """
    delta = {"grow": 1, "prune": -1}.get(move, 0)
    root_survives = not (move == "prune" and node == subtree_root)
    return _root_correction(len(state.tree), len(state.tree) + delta,
                            root_survives, subtree_root)


# ------------- a subtree root weighted towards smaller subtrees ------------- #
#
# Opt-in (a proposal's root_weight, gamma > 0): the root is drawn with
# probability w(rho) / W(T), where w(nu) = (n_nu + 1)^-gamma for the n_nu rows
# under nu, and W(T) is the sum of w over T's internal nodes. A move under rho
# leaves the rows reaching rho, and so w(rho), unchanged, so the auxiliary-
# variable correction log q(rho | T') - log q(rho | T) is log W(T) - log W(T').
# For an empty tree the draw is from {0} with probability one, so its "W" is
# w(0) itself. gamma = 0 is the uniform draw above (W = number of nodes).

def node_rows(state):
    """
    {node: rows under it} for every node of the tree, from the leaves' row
    counts. Index-only bookkeeping: no row's value is read.
    """
    leaf_idx = state.leaf_idx
    m = state.nodes
    n = {}
    stack = [(0, False)]
    while stack:
        v, done = stack.pop()
        if v not in m:
            n[v] = len(leaf_idx[v]) if v in leaf_idx else 0
        elif done:
            n[v] = n[int(m[v][1])] + n[int(m[v][2])]
        else:
            stack.append((v, True))
            stack.append((int(m[v][1]), False))
            stack.append((int(m[v][2]), False))
    return n


def _root_weights(state, gamma):
    """(pool, weights) of the weighted root draw; pool is [0] for an empty tree."""
    n = node_rows(state)
    pool = state.list_nodes or [0]
    return pool, (np.array([n[v] for v in pool], dtype=float) + 1.0) ** -gamma


def log_root_norm(state, gamma):
    """log W(T), memoised on the tree: a tree is never mutated once evaluated,
    and deep_copy does not carry the memo over."""
    memo = getattr(state, '_log_root_norm', None)
    if memo is not None and memo[0] == gamma:
        return memo[1]
    value = float(log(_root_weights(state, gamma)[1].sum()))
    state._log_root_norm = (gamma, value)
    return value


def draw_weighted_subtree(state, rng, gamma):
    """draw_subtree with the root drawn with probability w(rho) / W(T)."""
    pool, w = _root_weights(state, gamma)
    cum = np.cumsum(w)
    state._log_root_norm = (gamma, float(log(cum[-1])))
    i = int(np.searchsorted(cum, rng.nprng.random() * cum[-1], side='right'))
    root = pool[min(i, len(pool) - 1)]
    return root, make_context(state, root)


def weighted_root_correction(state, state_new, subtree_root, gamma):
    """root_correction for the weighted draw: log W(T) - log W(T'), and barred
    exactly when the uniform draw's is."""
    if state_new.tree:
        if subtree_root not in state_new.nodes:
            return large_neg
    elif subtree_root != 0:
        return large_neg
    return log_root_norm(state, gamma) - log_root_norm(state_new, gamma)


# ------------- the informed split proposal (problem.split_proposal) ------------- #
#
# For a grow or a change, draw a small subset S of the moved node's rows (with
# replacement, so its distribution depends only on the set of rows), score every
# feature by the information gain of its best split on S, and propose
#     feature ~ (1 - eps) softmax(eta * gain) + eps uniform,
#     threshold ~ (1 - eps_thr) uniform on that feature's best interval on S
#                 + eps_thr * the base threshold proposal.
# The mixture keeps every split's density positive. A move and its reverse
# draw S from the same set of rows (a grown leaf's rows are the new node's; a
# changed node keeps its rows), so for each S the pair is an exact MH step with
# the q ratio below, and mixing over S keeps it exact.

def informed_subset(problem, rows, rng):
    """
    S: problem.informed_rows draws, with replacement, from `rows`; or, with
    informed_rows = 0, all of `rows` (problem.informed_fraction = 1) or a random
    informed_fraction of them, without replacement (DA sets it to its ss_prop,
    so that its proposal sees as many rows as its screen). Either way S's
    distribution depends only on the set of rows. Index-only.
    """
    rows = np.asarray(rows)
    if len(rows) == 0:
        return rows
    if problem.informed_rows > 0:
        return rows[rng.nprng.integers(0, len(rows), problem.informed_rows)]
    frac = getattr(problem, "informed_fraction", 1.0)
    if frac >= 1.0:
        return rows
    k = max(2, int(round(frac * len(rows))))
    return rows[rng.nprng.choice(len(rows), min(k, len(rows)), replace=False)]


def _best_splits(problem, S):
    """
    (log feature weights, best interval (lo, hi) per feature or None) on S.
    Reads every feature and the label of each row of S.
    """
    F, K = problem.n_features, problem.num_classes
    if len(S) < 2:
        return np.full(F, -log(F)), [None] * F
    if len(S) > 2048:
        return _best_splits_large(problem, S)
    X = np.asarray(problem.X[S], dtype=float)
    y = problem.y[S]
    problem.reads.n += len(S) * (F + 1)
    n = len(S)
    order = np.argsort(X, axis=0, kind="stable")
    xs = np.take_along_axis(X, order, axis=0)
    onehot = (y[order][..., None] == np.arange(K)).astype(float)
    left = np.cumsum(onehot, axis=0)[:-1]
    total = onehot.sum(axis=0)
    right = total[None] - left
    nl = np.arange(1, n, dtype=float)[:, None]

    def n_entropy(c, m):
        with np.errstate(divide="ignore", invalid="ignore"):
            p = c / m[..., None]
            return -(m[..., None] * np.where(p > 0, p * np.log(p), 0.0)).sum(-1)
    gain = (n_entropy(total[None], np.full((1, F), float(n)))
            - n_entropy(left, nl) - n_entropy(right, n - nl))
    gain = np.where(xs[1:] > xs[:-1], gain, -np.inf)
    best = np.argmax(gain, axis=0)
    g = gain[best, np.arange(F)]
    ok = np.isfinite(g)
    w = np.full(F, 1.0 / F)
    if ok.any():
        z = problem.informed_eta * (g[ok] - g[ok].max())
        soft = np.zeros(F)
        soft[ok] = np.exp(z) / np.exp(z).sum()
        w = (1.0 - problem.informed_eps) * soft + problem.informed_eps / F
    intervals = [(float(xs[b, f]), float(xs[b + 1, f])) if ok[f] else None
                 for f, b in enumerate(best)]
    return np.log(w), intervals


def _lq_thr(problem, feat, thr, interval):
    """log q(thr | feat, S): the best-interval part mixed with the base proposal."""
    base = problem.lp_thr_proposal(feat, thr)
    if interval is None:
        return base
    lo, hi = interval
    e = problem.informed_eps_thr
    inside = (1.0 - e) / (hi - lo) if lo <= thr < hi else 0.0
    return log(inside + e * exp(base))


def _draw_informed(problem, log_w, intervals, rng):
    """(feat, thr, log q(feat, thr | S))."""
    w = np.exp(log_w)
    feat = int(rng.nprng.choice(problem.n_features, p=w / w.sum()))
    interval = intervals[feat]
    if interval is not None and rng.random() >= problem.informed_eps_thr:
        thr = interval[0] + rng.random() * (interval[1] - interval[0])
    else:
        thr, _ = problem.random_threshold(feat, rng)
    return feat, thr, log_w[feat] + _lq_thr(problem, feat, thr, interval)


# ------------- XBART's split rule as an MH proposal (problem.split_proposal) ------------- #
#
# He, Yalov and Hahn (2019), Section 3.1, Eq. 4: the variable and cutpoint are
# drawn jointly in proportion to exp(log marginal likelihood of the split) on
# all the node's rows, over the candidate cutpoints of their Section 3.3 (every
# j-th sorted value, j = (n_node - 2) // max(sqrt(n), 100)). Here the marginal
# likelihood is the model's own Dirichlet-Multinomial, the threshold is uniform
# in the gap after the chosen cutpoint, and informed_eps of the uniform
# proposal keeps every split (the starting trees' included) reversible.

def _xbart_splits(problem, rows):
    """Candidate gaps over all the node's rows: (feat, lo, hi, score) arrays and
    log of the normaliser sum(exp(score)). Reads every feature and label."""
    rows = np.asarray(rows)
    n, F, K = len(rows), problem.n_features, problem.num_classes
    problem.reads.n += n * (F + 1)
    empty = (np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0), -np.inf)
    m = max(problem.min_samples_leaf, 1)
    if n < 2 * m:
        return empty
    y = problem.y[rows]
    a0, alpha = problem._a0, problem.alpha
    total = np.bincount(y, minlength=K).astype(float)

    def log_dm(counts, size):
        return gammaln(a0) - gammaln(a0 + size) + gammaln(counts + alpha).sum(-1) - K * gammaln(alpha)
    parent = log_dm(total, float(n))
    C = max(int(np.sqrt(problem.n_rows)), 100)
    step = max(1, (n - 2) // C)
    fs, los, his, scores = [], [], [], []
    for f in range(F):
        x = np.asarray(problem.X[rows, f], dtype=float)
        order = np.argsort(x, kind="stable")
        xs = x[order]
        pos = np.arange(m - 1, n - m, step)          # cut after sorted position pos
        pos = pos[xs[pos] < xs[pos + 1]]
        if len(pos) == 0:
            continue
        onehot = np.zeros((n, K))
        onehot[np.arange(n), y[order]] = 1.0
        left = np.cumsum(onehot, axis=0)[pos]
        nl = (pos + 1).astype(float)
        score = log_dm(left, nl) + log_dm(total - left, n - nl) - parent
        fs.append(np.full(len(pos), f))
        los.append(xs[pos])
        his.append(xs[pos + 1])
        scores.append(score)
    if not fs:
        return empty
    feat, lo, hi, score = (np.concatenate(a) for a in (fs, los, his, scores))
    top = score.max()
    return feat, lo, hi, score, float(top + np.log(np.exp(score - top).sum()))


def _lq_xbart(problem, cands, feat, thr):
    """log q(feat, thr): the XBART part (the gap holding thr, if any) mixed with
    informed_eps of the uniform feature and base threshold proposal."""
    feats, lo, hi, score, log_z = cands
    eps = problem.informed_eps
    base = eps * exp(problem.lp_feats + problem.lp_thr_proposal(feat, thr))
    hit = np.flatnonzero((feats == feat) & (lo <= thr) & (thr < hi))
    part = 0.0
    if len(hit):
        j = hit[0]
        part = (1.0 - eps) * exp(score[j] - log_z) / (hi[j] - lo[j])
    return log(part + base)


def _draw_xbart(problem, cands, rng):
    feats, lo, hi, score, log_z = cands
    if len(feats) and rng.random() >= problem.informed_eps:
        p = np.exp(score - log_z)
        j = int(rng.nprng.choice(len(p), p=p / p.sum()))
        feat = int(feats[j])
        thr = lo[j] + rng.random() * (hi[j] - lo[j])
    else:
        feat = problem.random_feature(rng)
        thr, _ = problem.random_threshold(feat, rng)
    return feat, thr, _lq_xbart(problem, cands, feat, thr)


# ------------- the marginal-likelihood split proposal (problem.split_proposal) ------------- #
#
# Lakshminarayanan, Roy and Teh's (2013) optimal proposal, with the split
# likelihoods computed on a random subset S of the moved node's rows (drawn as
# for "informed"): every (feature, gap between adjacent distinct values on S)
# is weighted by exp(the Dirichlet-Multinomial marginal log-likelihood gain of
# cutting there), one is drawn, and the threshold is uniform in its gap; with
# probability informed_eps the uniform feature and base threshold proposal is
# used instead. Draw and density are _draw_xbart and _lq_xbart: only the
# candidates differ (all gaps on S, rather than XBART's thinned gaps on all rows).

def _marginal_splits(problem, S):
    """(feat, lo, hi, score, log_z) over every gap of every feature on S.
    Reads every feature and the label of each row of S."""
    S = np.asarray(S)
    n, F, K = len(S), problem.n_features, problem.num_classes
    empty = (np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0), -np.inf)
    if n < 2:
        return empty
    problem.reads.n += n * (F + 1)
    y = problem.y[S]
    a0, alpha = problem._a0, problem.alpha
    total = np.bincount(y, minlength=K).astype(float)

    def log_dm(counts, size):
        return gammaln(a0) - gammaln(a0 + size) + gammaln(counts + alpha).sum(-1) - K * gammaln(alpha)
    parent = log_dm(total, float(n))
    nl = np.arange(1, n, dtype=float)
    fs, los, his, scores = [], [], [], []
    # All features at once while that fits comfortably, else one at a time.
    chunk = F if n * F * K <= 4_000_000 else 1
    for f0 in range(0, F, chunk):
        X = np.asarray(problem.X[S, f0:f0 + chunk], dtype=float)
        order = np.argsort(X, axis=0, kind="stable")
        xs = np.take_along_axis(X, order, axis=0)
        onehot = (y[order][..., None] == np.arange(K)).astype(float)
        left = np.cumsum(onehot, axis=0)[:-1]                   # (n-1, chunk, K)
        score = (log_dm(left, nl[:, None]) + log_dm(total[None, None] - left, n - nl[:, None])
                 - parent)
        ok = xs[1:] > xs[:-1]                                    # a gap between distinct values
        b, j = np.nonzero(ok)
        fs.append(f0 + j)
        los.append(xs[b, j])
        his.append(xs[b + 1, j])
        scores.append(score[b, j])
    feat, lo, hi, score = (np.concatenate(a) for a in (fs, los, his, scores))
    if len(feat) == 0:
        return empty
    top = score.max()
    return feat, lo, hi, score, float(top + np.log(np.exp(score - top).sum()))


def _n_entropy(c, m):
    """m times the entropy of the class counts c (rows of c), in nats."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = c / m[..., None]
        return -(m[..., None] * np.where(p > 0, p * np.log(p), 0.0)).sum(-1)


def _best_splits_large(problem, S):
    """_best_splits one feature at a time, for an S too big to hold every
    feature's sorted class counts at once. The same result."""
    F, K = problem.n_features, problem.num_classes
    y = problem.y[S]
    problem.reads.n += len(S) * (F + 1)
    n = len(S)
    total = np.bincount(y, minlength=K).astype(float)
    parent = float(_n_entropy(total[None], np.array([float(n)]))[0])
    nl = np.arange(1, n, dtype=float)
    g = np.full(F, -np.inf)
    intervals = [None] * F
    for f in range(F):
        x = np.asarray(problem.X[S, f], dtype=float)
        order = np.argsort(x, kind="stable")
        xs = x[order]
        onehot = np.zeros((n, K))
        onehot[np.arange(n), y[order]] = 1.0
        left = np.cumsum(onehot, axis=0)[:-1]
        gain = parent - _n_entropy(left, nl) - _n_entropy(total[None] - left, n - nl)
        gain = np.where(xs[1:] > xs[:-1], gain, -np.inf)
        b = int(np.argmax(gain))
        if np.isfinite(gain[b]):
            g[f] = gain[b]
            intervals[f] = (float(xs[b]), float(xs[b + 1]))
    ok = np.isfinite(g)
    w = np.full(F, 1.0 / F)
    if ok.any():
        z = problem.informed_eta * (g[ok] - g[ok].max())
        soft = np.zeros(F)
        soft[ok] = np.exp(z) / np.exp(z).sum()
        w = (1.0 - problem.informed_eps) * soft + problem.informed_eps / F
    return np.log(w), intervals


# ------------- Chipman, George and McCulloch's (1998) node split values ------------- #
#
# The threshold is a cut between two adjacent distinct values that the moved
# node's own rows take for the feature, chosen uniformly over those cuts and
# placed uniformly inside the gap (every point of a gap splits the node's rows
# the same way, so this is their uniform choice among available split values,
# with a density). node_thr_eps of the uniform proposal over the training range
# keeps every threshold proposable, so splits made some other way (the prior's
# starting trees) can still be undone. A move and its reverse see the same rows
# at the node, so the reverse density is computed from the same values.

def _node_values(problem, rows, feat):
    """Sorted distinct values of `feat` over the node's rows (reads each row once)."""
    rows = np.asarray(rows)
    problem.reads.n += len(rows)
    return np.unique(np.asarray(problem.X[rows, feat], dtype=float))


def _lq_node_threshold(problem, values, feat, thr):
    """log q(thr | feat, node rows)."""
    base = problem.lp_thr_proposal(feat, thr)
    if len(values) < 2:
        return base
    eps = problem.node_thr_eps
    part = 0.0
    if values[0] <= thr < values[-1]:
        j = int(np.searchsorted(values, thr, side="right")) - 1
        part = (1.0 - eps) / (len(values) - 1) / (values[j + 1] - values[j])
    return log(part + eps * exp(base))


def _node_threshold(problem, rows, feat, rng):
    """(thr, log q(thr | feat, node rows))."""
    values = _node_values(problem, rows, feat)
    if len(values) >= 2 and rng.random() >= problem.node_thr_eps:
        j = int(rng.nprng.integers(0, len(values) - 1))
        thr = values[j] + rng.random() * (values[j + 1] - values[j])
    else:
        thr, _ = problem.random_threshold(feat, rng)
    return thr, _lq_node_threshold(problem, values, feat, thr)
