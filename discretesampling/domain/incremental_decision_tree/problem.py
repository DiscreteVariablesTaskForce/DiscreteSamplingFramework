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
    """

    def __init__(self, X, y, num_classes=None, alpha=1.0, lam=2.0,
                 min_samples_leaf=3, max_tree_size=None):
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
        Draw a threshold uniformly from the range of the given feature.
        """
        v_min, v_max = self.vals[int(feat)]
        return rng.uniform(v_min, v_max), self.lp_vals[int(feat)]

    def random_feature(self, rng):
        return int(rng.nprng.integers(0, self.n_features))

    def __repr__(self):
        return (f"<IncrementalTreeProblem {self.n_rows} rows, "
                f"{self.n_features} features, {self.num_classes} classes>")
