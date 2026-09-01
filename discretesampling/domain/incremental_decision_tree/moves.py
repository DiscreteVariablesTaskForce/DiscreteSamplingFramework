from collections import deque
from math import log

import numpy as np

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


def make_context(state, root):
    """
    One walk of the subtree under `root`, collecting everything a move needs.
    """
    if not state.tree:
        leaves = [root]
        leaf_idx = {leaf: state.leaf_idx[leaf] for leaf in leaves if leaf in state.leaf_idx}
        return SubtreeContext(root, [], [], leaves, leaf_idx)
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
    leaf_idx = {leaf: state.leaf_idx[leaf] for leaf in leaves if leaf in state.leaf_idx}
    return SubtreeContext(root, nodes, terminal_nodes, leaves, leaf_idx)


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


def change_admissible(state, node, feat, thr, idx):
    """
    Check whether a change move leaves every leaf under `node` still meets
    min_samples_leaf. This is the admissibility check for a change move only,
    because grow and prune are already screened by select_move.
    """
    m = state.nodes
    X = state.problem.X
    min_leaf = state.problem.min_samples_leaf
    queue = deque([(node, idx, True)])
    while queue:
        curr, curr_idx, is_changed = queue.popleft()
        row = m.get(curr)
        if row is None:
            if len(curr_idx) < min_leaf:
                return False
            continue
        curr_feat = int(feat) if is_changed else int(row[3])
        curr_thr = float(thr) if is_changed else float(row[4])
        left = X[curr_idx, curr_feat] < curr_thr
        queue.append((int(row[1]), curr_idx[left], False))
        queue.append((int(row[2]), curr_idx[~left], False))
    return True


def valid_threshold(problem, feat, thr, idx):
    """
    Check whether a threshold produces child leaves that meet min_samples_leaf.
    """
    left = np.count_nonzero(problem.X[idx, feat] < thr)
    return left >= problem.min_samples_leaf and (len(idx) - left) >= problem.min_samples_leaf


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

    if move == "grow":
        feat = problem.random_feature(rng)
        thr, lp_thr = problem.random_threshold(feat, rng)
        if not valid_threshold(problem, feat, thr, idx):
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
        prop_correction = log(q_b) - log(q_f) - lp_thr - problem.lp_feats

    elif move == "prune":
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
        prop_correction = log(q_b) + lp_old_thr + problem.lp_feats - log(q_f)

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

    return prop_correction, {'node': node, 'feat': feat, 'thr': thr, 'stump': stump}


def apply_subtree_proposal(state, ctx, move, prop_move):
    """
    Apply a move to the state, updating the context in place.
    Returns the new state, which is the same object as the input.
    """
    if move == "stay":
        return state
    node = prop_move['node']

    if move == "grow":
        state.grow(node, prop_move['feat'], prop_move['thr'])
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
        state.change(node, prop_move['feat'], prop_move['thr'])
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
