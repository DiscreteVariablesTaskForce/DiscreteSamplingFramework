import numpy as np
from math import log
from scipy.special import gammaln


# -np.inf keeps breaking log acceptance ratios and weight updates, so an
# out-of-support state gets a large negative number instead. Half of it is the
# test for "already dead": no combination of real log-densities gets near it.
large_neg = -1e9
BARRED = large_neg / 2


class IncrementalTreeProblem:
    """
    Data plus hyperparameters for one incremental decision tree run,
    contains all the information needed to evaluate a tree's log-target density.

    Parameters
    ----------
    X, y             : the training rows and their integer class labels
    num_classes      : inferred from y when not given
    alpha            : Dirichlet-Multinomial concentration, per class
    lam              : mean of the Poisson prior on the number of nodes
    min_samples_leaf : a leaf holding fewer rows than this is outside the
                       target's support (3 -- below that a leaf is empty for
                       practical purposes)
    max_tree_size    : node count at which grow moves stop being offered
    threshold_proposal : how split thresholds are PROPOSED -- "uniform" draws
                       from the feature's whole range, "data" draws an interval
                       between adjacent observed values and then a point inside
                       it, so proposals concentrate where the data is. This does
                       not change the target: the threshold PRIOR is uniform on
                       the range either way (see lp_vals vs lp_thr_proposal).
    """

    def __init__(self, X, y, num_classes=None, alpha=1.0, lam=2.0,
                 min_samples_leaf=3, max_tree_size=None,
                 threshold_proposal="uniform"):
        self.X = np.asfortranarray(X)
        self.y = np.ascontiguousarray(y).astype(np.int64, copy=False)
        self.num_classes = int(num_classes if num_classes is not None
                               else len(np.unique(self.y)))
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_tree_size = float('inf') if max_tree_size is None else float(max_tree_size)

        self.n_rows, self.n_features = self.X.shape

        # Dirichlet-Multinomial constants, evaluated on every leaf of every
        # proposal otherwise.
        self._a0 = self.alpha * self.num_classes
        self._lgamma_a0 = float(gammaln(self._a0))
        self._K_lgamma_alpha = self.num_classes * float(gammaln(self.alpha))

        # Splits are drawn uniformly: a feature from the pool, then a threshold
        # from that feature's global range. Both densities are constant, so
        # both are precomputed.
        self.feature_pool = list(range(self.n_features))
        self.lp_feats = -log(self.n_features)
        self.vals = []
        self.lp_vals = []
        for feat in self.feature_pool:
            v_min = float(np.min(self.X[:, feat]))
            v_max = float(np.max(self.X[:, feat]))
            if not v_max > v_min:
                raise ValueError(
                    f"feature {feat} is constant ({v_min}); drop invariant "
                    "features before building the problem")
            self.vals.append((v_min, v_max))
            self.lp_vals.append(-log(v_max - v_min))

        if threshold_proposal not in ("uniform", "data"):
            raise ValueError("threshold_proposal must be 'uniform' or 'data', "
                             f"got {threshold_proposal!r}")
        self.threshold_proposal = threshold_proposal
        self._intervals = {}

        self._block_of = None
        self._all_rows = None
        self._root_counts = None

    # ------------------------------------------------------------------ #

    def all_rows(self):
        """
        Returns a 1D array of all row indices in the dataset, from 0 to n_rows-1.
        """
        if self._all_rows is None:
            self._all_rows = np.arange(self.n_rows)
        return self._all_rows

    def root_counts(self):
        """Class counts over the whole dataset: every stump's only leaf."""
        if self._root_counts is None:
            self._root_counts = np.bincount(self.y, minlength=self.num_classes)
        return self._root_counts

    def block_buffer(self):
        """
        Returns a 1D array of length n_rows, filled with -1, for use as a
        temporary buffer when routing rows through the tree. The same buffer is
        reused across calls to route(X) to avoid repeated allocations.
        """
        buf = self._block_of
        if buf is None or len(buf) != self.n_rows:
            buf = self._block_of = np.full(self.n_rows, -1, dtype=np.int16)
        return buf

    def random_threshold(self, feat, rng):
        """
        Draw a threshold for the given feature from the PROPOSAL.
        Returns (threshold, log q(threshold | feature)).

        The returned density is the proposal's, not the prior's -- see
        lp_thr_proposal. They coincide for the default uniform proposal, but
        they are separate quantities and the MH correction needs this one.
        """
        feat = int(feat)
        if self.threshold_proposal == "data":
            edges, widths, log_widths = self._interval_cache(feat)
            j = int(rng.nprng.integers(0, len(widths)))
            return (edges[j] + rng.random() * widths[j],
                    -log(len(widths)) - log_widths[j])
        v_min, v_max = self.vals[feat]
        return rng.uniform(v_min, v_max), self.lp_vals[feat]

    def lp_thr_proposal(self, feat, thr):
        """
        log q(thr | feat): the density under which random_threshold would have
        drawn `thr`. This is the PROPOSAL density.

        Kept distinct from lp_vals, which is the threshold PRIOR the target
        integrates over. The two are equal only for the default uniform
        proposal. Reusing lp_vals for this role -- as the reverse corrections
        in moves.evaluate_subtree_move once did -- makes any change to the
        proposal silently break detailed balance, because the forward draw
        would be scored under one density and the reverse under another.
        """
        feat = int(feat)
        if self.threshold_proposal == "data":
            edges, widths, log_widths = self._interval_cache(feat)
            j = int(np.clip(np.searchsorted(edges, float(thr), side="right") - 1,
                            0, len(widths) - 1))
            return -log(len(widths)) - log_widths[j]
        return self.lp_vals[feat]

    def _interval_cache(self, feat):
        """
        The feature's observed values, as the edges of the intervals the "data"
        threshold proposal draws from: pick an interval uniformly, then a point
        uniformly inside it. Narrow intervals are where the data is dense, so
        this puts proposal mass where splits actually separate rows, while
        staying continuous on the prior's support.
        """
        cached = self._intervals.get(feat)
        if cached is None:
            edges = np.unique(np.asarray(self.X[:, feat], dtype=np.float64))
            widths = np.diff(edges)
            cached = (edges, widths, np.log(widths))
            self._intervals[feat] = cached
        return cached

    def random_feature(self, rng):
        return int(rng.nprng.integers(0, self.n_features))

    def __repr__(self):
        return (f"<IncrementalTreeProblem {self.n_rows} rows, "
                f"{self.n_features} features, {self.num_classes} classes>")
