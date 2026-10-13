import math
from collections import deque

import numpy as np

from discretesampling.base.random import RNG
from discretesampling.base.types import DiscreteVariableInitialProposal
from discretesampling.domain.incremental_decision_tree.incremental_tree import IncrementalTree


def tree_rows(rows, n_features=None):
    """
    A hand-written tree as the [node, left, right, feat, thr, depth] rows
    IncrementalTree.from_rows takes, or a ValueError naming what is wrong.

    `rows` is one [node, left, right, feat, thr] per internal node, in any
    order; a sixth entry, the depth IncrementalTree.tree carries, is accepted
    and recomputed. The root is node 0, a row goes left when
    X[row, feat] < thr, and an id that has no row of its own is a leaf.
    """
    try:
        nodes = {int(r[0]): (int(r[1]), int(r[2]), int(r[3]), float(r[4]))
                 for r in rows if len(r) in (5, 6)}
    except (TypeError, ValueError):
        nodes = None
    if not nodes or len(nodes) != len(rows):
        raise ValueError("a tree is a non-empty list of [node, left, right, feat, thr] "
                         "rows, one per internal node, each with its own node id")
    if 0 not in nodes:
        raise ValueError("the tree has no row for node 0, its root")
    children = [c for left, right, _, _ in nodes.values() for c in (left, right)]
    if len(set(children)) != len(children) or 0 in children:
        raise ValueError("every node other than the root 0 must be the child of "
                         "exactly one node")
    out, stack = [], [(0, 0)]
    while stack:
        node, depth = stack.pop()
        left, right, feat, thr = nodes[node]
        if not math.isfinite(thr):
            raise ValueError(f"node {node}: threshold {thr} is not finite")
        if feat < 0 or (n_features is not None and feat >= n_features):
            raise ValueError(f"node {node}: feature {feat} is out of range"
                             + ("" if n_features is None else
                                f" (the data has {n_features}, numbered from 0)"))
        out.append([node, left, right, feat, thr, depth])
        stack.extend((c, depth + 1) for c in (right, left) if c in nodes)
    if len(out) != len(nodes):
        lost = sorted(set(nodes) - {r[0] for r in out})
        raise ValueError(f"node(s) {lost} are not reachable from the root 0")
    return out


class IncrementalTreeInitialProposal(DiscreteVariableInitialProposal):
    """
    Initial proposal for the incremental decision tree domain: a point mass on
    one starting tree, so every chain and every particle starts from the same
    place.

    With n_nodes=0 (the default) that tree is a stump, i.e. a tree with no
    splits, a single leaf where all training data is routed.

    With n_nodes=N > 0 it is one random tree with exactly N internal nodes,
    drawn once from RNG(seed) and never from the rng a sampler passes in. It
    depends on nothing but the data, min_samples_leaf, N and seed, so it is the
    same tree in every process, for every sampler and under every proposal:
    - its shape is drawn uniformly from the Catalan(N) binary tree shapes,
      as the target's structural prior is at a fixed size,
    - its splits are filled in from the root down, each node's left subtree
      size drawn first. A node then draws a feature uniformly from those it
      can split leaving each side min_samples_leaf rows for every leaf the
      shape hangs below it, then one of that feature's valid split points --
      the gaps between adjacent distinct values of its rows that do so --
      uniformly, then a threshold uniformly inside that gap. So the tree is
      inside the target's support, its splits are spread over the rows rather
      than over the feature's range (which would mostly shave a few rows off
      one end), and a deep subtree is never starved of rows.
    A draw that reaches a node with no feature it can split -- possible only
    through tied values -- is started again from scratch.

    With tree=rows it is that tree and nothing is drawn: see tree_rows for
    the rows. It is refused if it is outside the target's support, i.e. if
    any of its leaves holds fewer than min_samples_leaf rows.
    """

    def __init__(self, problem, n_nodes=0, seed=0, max_attempts=1000, tree=None):
        if tree is not None:
            tree = tree_rows(tree, problem.n_features)
            if n_nodes not in (0, len(tree)):
                raise ValueError(f"n_nodes={n_nodes} given with a tree of "
                                 f"{len(tree)} internal nodes: set one of the two")
            n_nodes = len(tree)
        self._rows = tree
        if n_nodes < 0:
            raise ValueError(f"n_nodes must be >= 0, got {n_nodes}")
        if n_nodes > problem.max_tree_size:
            raise ValueError(f"n_nodes={n_nodes} is above max_tree_size="
                             f"{problem.max_tree_size:g}")
        if (n_nodes + 1) * max(problem.min_samples_leaf, 1) > problem.n_rows:
            raise ValueError(f"n_nodes={n_nodes} needs {n_nodes + 1} leaves of "
                             f"min_samples_leaf={problem.min_samples_leaf} rows, "
                             f"more than the {problem.n_rows} there are")
        self.problem = problem
        self.n_nodes = int(n_nodes)
        self.seed = seed
        self.max_attempts = int(max_attempts)
        # Exact Catalan numbers, for the left-subtree size draws.
        self._catalan = [1]
        for n in range(self.n_nodes):
            self._catalan.append(self._catalan[-1] * 2 * (2 * n + 1) // (n + 2))
        # Built on first use, so a proposal that is never sampled costs nothing.
        self._start = None

    # ------------------- valid splits ------------------- #

    def _needs(self, k_left, k_right):
        """Rows each side of a split must keep for subtrees with k_left and
        k_right internal nodes below it: min_samples_leaf for every leaf."""
        m = max(self.problem.min_samples_leaf, 1)
        return (k_left + 1) * m, (k_right + 1) * m

    def _features(self, idx, need_left, need_right):
        """The features some threshold splits the rows `idx` on with at least
        need_left rows below it and need_right at or above it."""
        problem = self.problem
        n = len(idx)
        if n < need_left + need_right:
            return np.empty(0, dtype=np.int64)
        # A split exists iff the need_left-th smallest value is below the
        # need_right-th largest: a threshold just above the first does it.
        lo, hi = need_left - 1, n - need_right
        valid = []
        for f in range(problem.n_features):
            part = np.partition(problem.X[idx, f], (lo, hi))
            if part[lo] < part[hi]:
                valid.append(f)
        return np.asarray(valid, dtype=np.int64)

    def _gaps(self, idx, feat, need_left, need_right):
        """
        (lo, width) of every valid split point of the rows `idx` on `feat`:
        the gaps (u_i, u_{i+1}] between adjacent distinct values that leave at
        least need_left rows below and need_right above, ascending.
        """
        n = len(idx)
        u, counts = np.unique(self.problem.X[idx, feat], return_counts=True)
        below = np.cumsum(counts)[:-1]           # rows <= u_i, for each gap i
        ok = (below >= need_left) & (n - below >= need_right)
        return u[:-1][ok], np.diff(u)[ok]

    def _left_size(self, n, rng):
        """Internal nodes in the left subtree of one with n, drawn so that the
        whole shape is uniform: k with probability C_k C_{n-1-k} / C_n."""
        cat = self._catalan
        u = rng.random() * cat[n]
        acc = 0
        for k in range(n - 1):
            acc += cat[k] * cat[n - 1 - k]
            if u < acc:
                return k
        return n - 1

    # ------------------- the starting tree ------------------- #

    def start(self):
        """The one tree every sample is a copy of."""
        if self._start is None:
            self._start = self._draw()
        return self._start

    def _draw(self):
        if self._rows is not None:
            return self._given()
        if self.n_nodes == 0:
            return IncrementalTree.stump(self.problem)
        rng = RNG(self.seed)
        for _ in range(self.max_attempts):
            x = self._attempt(rng)
            if x is not None:
                return x
        raise RuntimeError(
            f"no random tree with {self.n_nodes} internal nodes found in "
            f"{self.max_attempts} attempts: tied values leave some node no "
            f"split with min_samples_leaf={self.problem.min_samples_leaf} rows "
            f"for every leaf below it")

    def _given(self):
        """The tree handed in, with the data routed through it."""
        x = IncrementalTree.from_rows(self.problem, self._rows)
        m = self.problem.min_samples_leaf
        short = {leaf: len(rows) for leaf, rows in x.leaf_idx.items() if len(rows) < m}
        if short:
            raise ValueError(f"the given tree is outside the target's support: "
                             f"leaves {short} (leaf: rows) hold fewer than "
                             f"min_samples_leaf={m} rows")
        return x

    def _attempt(self, rng):
        """One top-down draw, or None if it reaches a node it cannot split."""
        x = IncrementalTree.stump(self.problem)
        queue = deque([(0, self.n_nodes)])
        while queue:
            leaf, n = queue.popleft()
            if n == 0:
                continue
            # The shape first: the split must leave each side rows enough
            # for every leaf of the subtree that hangs off it.
            k = self._left_size(n, rng)
            needs = self._needs(k, n - 1 - k)
            idx = x.leaf_idx[leaf]
            feats = self._features(idx, *needs)
            if feats.size == 0:
                return None
            feat = int(feats[rng.nprng.integers(0, feats.size)])
            lo, width = self._gaps(idx, feat, *needs)
            j = int(rng.nprng.integers(0, lo.size))
            # hi - width * [0, 1) is the half-open (lo, hi] the gap is.
            thr = float(lo[j] + width[j] * (1.0 - rng.random()))
            if not thr > lo[j]:
                # Rounded onto lo: too narrow a gap to draw inside.
                return None
            x.grow(leaf, feat, thr)
            L, R = x._child_nodes(leaf)
            queue.append((L, k))
            queue.append((R, n - 1 - k))
        return x

    # ------------------- the proposal ------------------- #

    def sample(self, rng=RNG(), target=None):
        # A copy each time: the rows are shared, the tree is not, so no two
        # particles hold the same object.
        return self.start().deep_copy()

    def eval(self, x, target=None):
        if len(x.tree) != self.n_nodes:
            return -math.inf
        if self.n_nodes == 0 and self._rows is None:
            return 0.0
        return 0.0 if x == self.start() else -math.inf

    def __repr__(self):
        return f"<IncrementalTreeInitialProposal {self.problem} n_nodes={self.n_nodes}>"
