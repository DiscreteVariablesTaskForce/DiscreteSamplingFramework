import math
from collections import deque

import numpy as np
from scipy.stats import poisson

from discretesampling.base.random import RNG
from discretesampling.base.types import DiscreteVariableInitialProposal
from discretesampling.domain.incremental_decision_tree.incremental_tree import IncrementalTree
from discretesampling.domain.incremental_decision_tree.target import IncrementalTreeTarget


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
    Initial proposal for the incremental decision tree domain.

    By default (n_nodes=None, tree=None) every sample is a fresh tree, drawn
    with the rng the sampler passes in, from the target's prior conditioned
    split by split on the tree staying admissible, i.e. on every leaf holding
    at least min_samples_leaf rows (m below):
    - its size n is the prior's Poisson(lam), truncated to the sizes an
      admissible tree can have: at most max_tree_size, and at most
      n_rows // m - 1, so that n + 1 leaves of m rows fit in the data,
    - its shape is the prior's, uniform over the Catalan(n) shapes,
    - its splits are filled in from the root down, each node's left subtree
      size drawn first. A node's split is the prior's -- a uniform feature,
      then a threshold uniform over that feature's whole range -- conditioned
      on leaving each side m rows for every leaf the shape hangs below it. So
      a feature is drawn with probability proportional to the fraction of its
      range where a threshold does that, and the threshold uniformly inside
      that part of the range.
    With m = 0 nothing is conditioned and this is exactly the prior.
    Otherwise the draw differs from the prior only by, at each internal node
    v, Z_v: the prior probability that v's split keeps the tree admissible,
    given the rows reaching v and the shape below it. eval returns this draw's
    own log-density, log p(T) - sum_v log Z_v, so the SMC starting weight
    target - eval is the log-likelihood plus sum_v log Z_v: an exact
    importance weight for the posterior. Drawing from the prior exactly and
    rejecting inadmissible trees gives the likelihood alone, but keeps around
    one draw in 300 to 2000 on the datasets used here, and none on covtype;
    Subsampling_Trees.tex, "Initialising the particles", has the argument.
    eval ignores the `target` it is handed: an importance weight needs the
    density the tree was actually drawn from, which is this one even when the
    target's prior differs.
    A draw that reaches a node no split can satisfy -- possible only through
    tied values -- is started again from scratch, size included.

    The other starts are a point mass on one starting tree, so every chain and
    every particle starts from the same place.

    With n_nodes=0 that tree is a stump, i.e. a tree with no splits, a single
    leaf where all training data is routed.

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

    def __init__(self, problem, n_nodes=None, seed=0, max_attempts=1000, tree=None):
        if tree is not None:
            tree = tree_rows(tree, problem.n_features)
            if n_nodes not in (None, 0, len(tree)):
                raise ValueError(f"n_nodes={n_nodes} given with a tree of "
                                 f"{len(tree)} internal nodes: set one of the two")
            n_nodes = len(tree)
        self._rows = tree
        self.problem = problem
        self.seed = seed
        self.max_attempts = int(max_attempts)
        self.from_prior = n_nodes is None
        self.n_nodes = n_nodes
        # Exact Catalan numbers, for the left-subtree size draws.
        self._catalan = [1]
        if self.from_prior:
            self._init_prior()
            return
        if n_nodes < 0:
            raise ValueError(f"n_nodes must be >= 0, got {n_nodes}")
        if n_nodes > problem.max_tree_size:
            raise ValueError(f"n_nodes={n_nodes} is above max_tree_size="
                             f"{problem.max_tree_size:g}")
        if (n_nodes + 1) * max(problem.min_samples_leaf, 1) > problem.n_rows:
            raise ValueError(f"n_nodes={n_nodes} needs {n_nodes + 1} leaves of "
                             f"min_samples_leaf={problem.min_samples_leaf} rows, "
                             f"more than the {problem.n_rows} there are")
        self.n_nodes = int(n_nodes)
        self._extend_catalan(self.n_nodes)
        # Built on first use, so a proposal that is never sampled costs nothing.
        self._start = None

    def _init_prior(self):
        problem = self.problem
        m = problem.min_samples_leaf
        if problem.n_rows < max(m, 1):
            raise ValueError(f"no tree is admissible: even a stump's one leaf "
                             f"needs min_samples_leaf={m} of the "
                             f"{problem.n_rows} rows")
        # The sizes an admissible tree can have; inf when nothing bounds it.
        self.max_nodes = problem.max_tree_size
        if m > 0:
            self.max_nodes = min(self.max_nodes, problem.n_rows // m - 1)
        self._cdf_max = float(poisson.cdf(self.max_nodes, problem.lam))
        self._vmin = np.array([lo for lo, _ in problem.vals])
        self._vmax = np.array([hi for _, hi in problem.vals])
        self._range = self._vmax - self._vmin
        # The prior is the target's own, so that target - eval leaves exactly
        # the likelihood and the per-split factors.
        self._prior = IncrementalTreeTarget(problem)

    def _extend_catalan(self, n):
        cat = self._catalan
        for k in range(len(cat) - 1, n):
            cat.append(cat[-1] * 2 * (2 * k + 1) // (k + 2))

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
        # u * C_n in exact integers: rng.random() is a multiple of 2**-53, and
        # C_n overflows a float from n ~ 500.
        u = (int(rng.random() * 2 ** 53) * cat[n]) >> 53
        acc = 0
        for k in range(n - 1):
            acc += cat[k] * cat[n - 1 - k]
            if u < acc:
                return k
        return n - 1

    # ------------------- the prior, split by split ------------------- #

    def _prior_size(self, rng):
        """n ~ Poisson(lam) truncated to n <= max_nodes, by inverting its CDF."""
        # u in (0, 1], so the quantile is in (0, F(max_nodes)] and n >= 0.
        q = (1.0 - rng.random()) * self._cdf_max
        n = poisson.ppf(q, self.problem.lam)
        return int(min(max(n, 0), self.max_nodes))

    def _valid_ranges(self, idx, k_left, k_right):
        """
        (lo, hi) per feature: the thresholds (lo, hi] that split the rows `idx`
        leaving min_samples_leaf rows for every leaf of a k_left-node subtree
        below and of a k_right-node one at or above; lo == hi where no
        threshold does. With min_samples_leaf=0 that is the whole prior range.
        """
        problem = self.problem
        m = problem.min_samples_leaf
        if m == 0:
            return self._vmin, self._vmax
        need_left, need_right = (k_left + 1) * m, (k_right + 1) * m
        n = len(idx)
        lo = np.zeros(problem.n_features)
        hi = np.zeros(problem.n_features)
        if n >= need_left + need_right:
            # Valid iff the need_left-th smallest value is below the
            # threshold and the need_right-th largest is at or above it.
            a, b = need_left - 1, n - need_right
            for f in range(problem.n_features):
                part = np.partition(problem.X[idx, f], (a, b))
                lo[f], hi[f] = part[a], part[b]
        return lo, hi

    def _valid_fractions(self, lo, hi):
        """The fraction of each feature's prior range that is valid."""
        return (hi - lo) / self._range

    def _log_z(self, frac):
        """log Z_v: the prior probability, 1/J * sum_f frac_f, that a split
        drawn from the prior has a threshold in its valid range."""
        return math.log(float(frac.sum()) / self.problem.n_features)

    def _prior_split(self, idx, k_left, k_right, rng):
        """(feat, thr, log Z_v) drawn from the prior conditioned on the split
        being valid, or None if no split is."""
        lo, hi = self._valid_ranges(idx, k_left, k_right)
        frac = self._valid_fractions(lo, hi)
        cum = np.cumsum(frac)
        if not cum[-1] > 0:
            return None
        feat = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        feat = min(feat, int(np.flatnonzero(frac)[-1]))
        # hi - width * [0, 1) is the half-open (lo, hi] the valid range is,
        # and never rounds above hi, which would cost the right side a row.
        thr = float(hi[feat] - (hi[feat] - lo[feat]) * rng.random())
        if not thr > lo[feat]:
            return None      # rounded onto lo: too narrow a range to draw inside
        return feat, thr, self._log_z(frac)

    def _prior_attempt(self, rng):
        """One top-down draw, or None if it reaches a node it cannot split."""
        n = self._prior_size(rng)
        self._extend_catalan(n)
        x = IncrementalTree.stump(self.problem)
        log_z = 0.0
        queue = deque([(0, n)])
        while queue:
            leaf, k = queue.popleft()
            if k == 0:
                continue
            k_left = self._left_size(k, rng)
            split = self._prior_split(x.leaf_idx[leaf], k_left, k - 1 - k_left, rng)
            if split is None:
                return None
            feat, thr, lz = split
            log_z += lz
            x.grow(leaf, feat, thr)
            left, right = x._child_nodes(leaf)
            queue.append((left, k_left))
            queue.append((right, k - 1 - k_left))
        # Memoised for eval, which SMC calls on this very object; a move works
        # on a deep_copy, which does not carry it.
        x._log_q0 = self._prior.evaluatePrior(x) - log_z
        return x

    def _draw_from_prior(self, rng):
        for _ in range(self.max_attempts):
            x = self._prior_attempt(rng)
            if x is not None:
                return x
        raise RuntimeError(
            f"no tree drawn from the prior in {self.max_attempts} attempts: "
            f"tied values leave some node no split with "
            f"min_samples_leaf={self.problem.min_samples_leaf} rows for every "
            f"leaf below it")

    def _prior_log_density(self, x):
        """log p(x) - sum_v log Z_v, recomputed from the tree itself."""
        problem = self.problem
        if len(x.tree) > self.max_nodes:
            return -math.inf
        m = problem.min_samples_leaf
        if any(len(rows) < m for rows in x.leaf_idx.values()):
            return -math.inf
        log_z = 0.0
        if m > 0:
            nodes = x.nodes
            # Internal nodes below each node, itself included: children first.
            order, stack = [], [0]
            while stack:
                node = stack.pop()
                if node in nodes:
                    order.append(node)
                    stack.extend((int(nodes[node][1]), int(nodes[node][2])))
            size = {}
            for node in reversed(order):
                left, right = int(nodes[node][1]), int(nodes[node][2])
                size[node] = 1 + size.get(left, 0) + size.get(right, 0)
            # Every leaf holds m rows, so every split is valid and Z_v > 0.
            stack = [(0, problem.all_rows())]
            while stack:
                node, idx = stack.pop()
                row = nodes.get(node)
                if row is None:
                    continue
                left, right = int(row[1]), int(row[2])
                lo, hi = self._valid_ranges(idx, size.get(left, 0), size.get(right, 0))
                log_z += self._log_z(self._valid_fractions(lo, hi))
                go_left = problem.X[idx, int(row[3])] < float(row[4])
                stack.append((left, idx[go_left]))
                stack.append((right, idx[~go_left]))
        return self._prior.evaluatePrior(x) - log_z

    # ------------------- the starting tree ------------------- #

    def start(self):
        """The one tree every sample is a copy of."""
        if self.from_prior:
            raise ValueError("drawn from the prior, every sample is a tree of "
                             "its own: there is no one starting tree")
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
        # Drawing a start is set-up, not sampling: the rows it reads are left
        # out of the problem's RowReads, which measures the sampler's cost.
        reads = self.problem.reads
        before = reads.n
        if self.from_prior:
            x = self._draw_from_prior(rng)
        else:
            # A copy each time: the rows are shared, the tree is not, so no
            # two particles hold the same object.
            x = self.start().deep_copy()
        reads.n = before
        return x

    def eval(self, x, target=None):
        if self.from_prior:
            cached = getattr(x, '_log_q0', None)
            if cached is not None and x.problem is self.problem:
                return cached
            return self._prior_log_density(x)
        if len(x.tree) != self.n_nodes:
            return -math.inf
        if self.n_nodes == 0 and self._rows is None:
            return 0.0
        return 0.0 if x == self.start() else -math.inf

    def __repr__(self):
        start = "from the prior" if self.from_prior else f"n_nodes={self.n_nodes}"
        return f"<IncrementalTreeInitialProposal {self.problem} {start}>"
