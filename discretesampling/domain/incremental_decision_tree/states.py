"""
Flat, array-backed storage for the tree states a run visits.

The samplers record what they *did* -- the tree at every MCMC iteration, and
every particle at every SMC step -- and nothing about how good it was. The
predictive metrics are computed afterwards, off these stored states, by
examples/incremental_decision_tree/evaluate_results.py. That split is what makes a long run affordable:
one evaluation routes every row of a split through a tree and costs far more
than the sampler iteration that produced it, so evaluating inline makes the
diagnostics the run, and fixes forever which metrics you get.

Storing every state naively would cost more than the metrics did. Two things
keep it small, and both are exact rather than a thinning:

  * Only *distinct* states are stored. A rejected MCMC move leaves the chain
    holding the object it already had, and SMC resampling puts one particle
    object in many slots, so the states visited are far fewer than the states
    recorded. What is stored per record is a slot index per particle.
  * A state is its tree rows plus its leaf class counts -- not its leaf
    membership, which is a row index per training row per leaf and is what
    makes a live IncBDTree big. Metrics need the counts and the splits; they
    re-route the split being evaluated themselves.

So the layout is a handful of flat arrays plus per-state offsets, and
`series.state(slot)` hands back a view that quacks like a tree:

    state.tree            rows [id, left, right, feat, thr, depth]
    len(state.tree)       O(1), does not build the rows
    state.counts[leaf]    class counts for one leaf
    state.counts.items()  / .values() / .keys()
    state.route(X)        the leaf each row of X reaches
    state.num_classes, state.alpha

No per-leaf Python objects are created unless something actually asks for a
leaf.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from discretesampling.domain.incremental_decision_tree.metrics import route_rows

TREE_COLUMNS = ('id', 'left', 'right', 'feat', 'thr', 'depth')

# The column dtypes. `thr` stays float64: it is compared against feature values
# to route rows, and a float32 round trip moves a threshold across a data point
# often enough to change which leaf a row lands in -- which would make a metric
# read back off disk differ from the one the live tree would have given.
_TREE_DTYPES = (('id', np.int32), ('left', np.int32), ('right', np.int32),
                ('feat', np.int32), ('thr', np.float64), ('depth', np.int32))


class TreeRows(Sequence):
    """
    The [id, left, right, feat, thr, depth] rows of one stored state.

    Held column-wise at each column's own width; the (n, 6) block callers
    expect is built on first element access and cached, so `len(state.tree)`
    stays free.
    """
    __slots__ = ('_cols', '_start', '_n', '_block')

    def __init__(self, cols, start, n):
        self._cols = cols
        self._start = int(start)
        self._n = int(n)
        self._block = None

    def _materialise(self):
        if self._block is None:
            a, b = self._start, self._start + self._n
            block = np.empty((self._n, 6), dtype=np.float64)
            for j, name in enumerate(TREE_COLUMNS):
                block[:, j] = self._cols[name][a:b]
            self._block = block
        return self._block

    def __len__(self):
        return self._n

    @property
    def shape(self):
        return (self._n, 6)

    def __getitem__(self, key):
        return self._materialise()[key]

    def __iter__(self):
        return iter(self._materialise())

    def __array__(self, dtype=None, copy=None):
        block = self._materialise()
        return block if dtype is None else block.astype(dtype)

    def __repr__(self):
        return f"<TreeRows {self._n} nodes>"


class LeafCounts(Mapping):
    """
    {leaf id: class counts} over two array slices, with no dict and no per-leaf
    ndarray objects. Lookup is a binary search on the sorted ids.
    """
    __slots__ = ('_ids', '_counts')

    def __init__(self, ids, counts):
        self._ids = ids
        self._counts = counts

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, leaf_id):
        pos = int(np.searchsorted(self._ids, leaf_id))
        if pos >= len(self._ids) or self._ids[pos] != leaf_id:
            raise KeyError(leaf_id)
        return self._counts[pos]

    def __iter__(self):
        return iter(self._ids.tolist())

    def __contains__(self, leaf_id):
        pos = int(np.searchsorted(self._ids, leaf_id))
        return pos < len(self._ids) and self._ids[pos] == leaf_id

    def keys(self):
        return self._ids.tolist()

    def values(self):
        # Straight over the rows; the Mapping default would binary-search every
        # key, and iterating the values is what metrics.py does.
        return iter(self._counts)

    def items(self):
        return zip(self._ids.tolist(), self._counts)

    @property
    def ids(self):
        return self._ids

    @property
    def block(self):
        """The (n_leaves, num_classes) counts block, for vectorised work."""
        return self._counts

    def __repr__(self):
        return f"<LeafCounts {len(self._ids)} leaves>"


class TreeStoreView:
    """One stored state, shaped like a tree as far as metrics.py is concerned."""
    __slots__ = ('tree', 'counts', 'num_classes', 'alpha')

    def __init__(self, tree, counts, num_classes, alpha):
        self.tree = tree
        self.counts = counts
        self.num_classes = num_classes
        self.alpha = alpha

    @property
    def leafs(self):
        return self.counts.keys()

    def route(self, X):
        """The leaf id each row of X reaches."""
        return route_rows(self.tree, X)

    def __repr__(self):
        return (f"<TreeStoreView {len(self.tree)} nodes, "
                f"{len(self.counts)} leaves>")


class StateSeries:
    """
    A run's recorded states: the distinct trees as flat arrays, plus an
    (n_records, n_particles) index saying which of them each particle held at
    each recorded iteration or step.

    MCMC records one state per iteration, so n_particles is 1. SMC records the
    whole population, so a record is the ensemble a metric is averaged over --
    with `log_weights` alongside, since the SMC estimator is the weighted one.

    Indexing costs only the small view objects; nothing per leaf is allocated
    until a leaf is asked for.
    """

    def __init__(self, cols, tree_lengths, leaf_lengths, leaf_ids, leaf_counts,
                 index, num_classes, alpha=1.0, log_weights=None,
                 record_iters=None):
        self._cols = {k: np.asarray(v) for k, v in cols.items()}
        self._tree_lengths = np.asarray(tree_lengths, dtype=np.int64)
        self._leaf_lengths = np.asarray(leaf_lengths, dtype=np.int64)
        self._leaf_ids = np.asarray(leaf_ids)
        self._leaf_counts = np.asarray(leaf_counts)
        index = np.asarray(index, dtype=np.int64)
        # MCMC records one state per iteration and may hand it over flat; a
        # record is an ensemble either way, so it is held two-dimensional.
        self._index = index.reshape(-1, 1) if index.ndim == 1 else index
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.log_weights = None if log_weights is None else np.asarray(log_weights)
        self._record_iters = (None if record_iters is None
                              else np.asarray(record_iters, dtype=np.int64))
        self._tree_offsets = np.concatenate(
            [[0], np.cumsum(self._tree_lengths, dtype=np.int64)])
        self._leaf_offsets = np.concatenate(
            [[0], np.cumsum(self._leaf_lengths, dtype=np.int64)])

    # ---------------- shape ---------------- #

    def __len__(self):
        """Recorded iterations / steps."""
        return len(self._index)

    @property
    def n_particles(self):
        return self._index.shape[1]

    @property
    def n_stored(self):
        """Distinct states actually held, which is what the arrays cost."""
        return len(self._tree_lengths)

    @property
    def node_counts(self):
        """Node count per *stored* state, without materialising any of them."""
        return self._tree_lengths

    @property
    def leaf_counts_per_state(self):
        """Leaf count per stored state."""
        return self._leaf_lengths

    def record_iterations(self):
        """
        The sampler iteration / step behind each record. Recorded explicitly
        rather than derived, so a run stored every k-th iteration still says
        which ones they were.
        """
        if self._record_iters is not None:
            return self._record_iters
        return np.arange(len(self), dtype=np.int64)

    # ---------------- states ---------------- #

    def state(self, slot):
        """One stored state, by slot -- what the index array holds."""
        slot = int(slot)
        if not 0 <= slot < self.n_stored:
            raise IndexError(f"state slot {slot} out of range for "
                             f"{self.n_stored} stored states")
        a = int(self._tree_offsets[slot])
        t_len = int(self._tree_lengths[slot])
        lo = int(self._leaf_offsets[slot])
        l_len = int(self._leaf_lengths[slot])
        return TreeStoreView(TreeRows(self._cols, a, t_len),
                             LeafCounts(self._leaf_ids[lo:lo + l_len],
                                        self._leaf_counts[lo:lo + l_len]),
                             self.num_classes, self.alpha)

    def slots(self, i):
        """The stored-state slot each particle held at record i."""
        return self._index[i]

    def __getitem__(self, i):
        """The ensemble at record i, as a list of views (one for MCMC)."""
        return [self.state(s) for s in self._index[i]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ---------------- weights ---------------- #

    def weights(self, i):
        """
        Normalised linear particle weights at record i, or None when the
        record is unweighted (MCMC, or an SMC run stored without weights).
        """
        if self.log_weights is None:
            return None
        logw = np.asarray(self.log_weights[i], dtype=np.float64)
        logw = logw - logw.max()
        w = np.exp(logw)
        total = w.sum()
        return w / total if total > 0 else np.full(len(w), 1.0 / len(w))

    def ensemble(self, i):
        """
        (slots, weights) for record i, deduplicated: a state held by several
        particles appears once, carrying their combined weight.

        Resampling puts one particle object in many slots, so an SMC record of
        500 particles routinely holds a handful of distinct trees. Evaluating
        each of them once instead of once per slot is exact, not an
        approximation, and is most of what makes evaluating a stored run
        cheaper than evaluating it inline was.

        Slots rather than views, because they are also the identity of the
        ensemble: a caller evaluating a whole run can key a cache on them and
        skip a record that repeats one it has already done.
        """
        slots = self._index[i]
        weights = self.weights(i)
        if weights is None:
            weights = np.full(len(slots), 1.0 / len(slots))
        uniq, inverse = np.unique(slots, return_inverse=True)
        combined = np.bincount(inverse, weights=weights, minlength=len(uniq))
        return uniq, combined

    def nbytes(self):
        total = sum(c.nbytes for c in self._cols.values())
        total += self._tree_lengths.nbytes + self._leaf_lengths.nbytes
        total += self._leaf_ids.nbytes + self._leaf_counts.nbytes
        total += self._index.nbytes
        if self.log_weights is not None:
            total += self.log_weights.nbytes
        return total

    def __repr__(self):
        return (f"<StateSeries {len(self)} records x {self.n_particles} "
                f"particles, {self.n_stored} distinct states, "
                f"{self.nbytes() / 1e6:.1f} MB>")


class _GrowableColumns:
    """Doubling flat buffers, one per named column."""

    def __init__(self, specs, capacity):
        self._cap = max(int(capacity), 1)
        self._n = 0
        self._cols = {name: np.empty(self._cap, dtype=dt) for name, dt in specs}

    def _reserve(self, extra):
        need = self._n + extra
        if need <= self._cap:
            return
        while self._cap < need:
            self._cap *= 2
        for name, col in self._cols.items():
            grown = np.empty(self._cap, dtype=col.dtype)
            grown[:self._n] = col[:self._n]
            self._cols[name] = grown

    def extend(self, block):
        """block: {name: 1-D array}, all of the same length."""
        k = len(next(iter(block.values())))
        self._reserve(k)
        for name, values in block.items():
            self._cols[name][self._n:self._n + k] = values
        self._n += k

    def arrays(self):
        return {name: col[:self._n] for name, col in self._cols.items()}


class _GrowableCounts:
    """A doubling (n, num_classes) int32 block."""

    def __init__(self, num_classes, capacity):
        self._k = max(int(num_classes), 1)
        self._cap = max(int(capacity), 1)
        self._n = 0
        self._buf = np.empty((self._cap, self._k), dtype=np.int32)

    def extend(self, block):
        k = len(block)
        if self._n + k > self._cap:
            while self._cap < self._n + k:
                self._cap *= 2
            grown = np.empty((self._cap, self._k), dtype=np.int32)
            grown[:self._n] = self._buf[:self._n]
            self._buf = grown
        self._buf[self._n:self._n + k] = block
        self._n += k

    def array(self):
        return self._buf[:self._n]


class StateRecorder:
    """
    Records the states a run visits straight into flat buffers.

    `record(states)` takes the whole ensemble at one iteration or step -- a
    one-element sequence for MCMC, the particle list for SMC -- and appends a
    row of slot indices. A state already stored is not stored again.

    Sameness is object identity, which here is exact and free. None of the
    proposals mutates the state it is handed: a rejected or barred move returns
    the very object it was given (IncrementalTreeProposalBase._stay), an
    accepted one returns a fresh deep_copy, and resampling redistributes
    references rather than copying. So two slots holding the same object hold
    the same tree, and two records holding the same object are the same state.

    To make identity safe, the recorder holds a reference to every object whose
    id it has in its tables -- an id belonging to a dead object could otherwise
    be reused by a live one and match the wrong slot. Only the current and the
    previous record's distinct states are held, which is the window that
    catches both kinds of repeat (aliasing within a record, persistence across
    one) and is a generation of particles the sampler is holding anyway.
    """

    def __init__(self, num_classes, alpha=1.0, capacity=1024):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self._tree = _GrowableColumns(_TREE_DTYPES, capacity * 8)
        self._leaf = _GrowableColumns((('ids', np.int32),), capacity * 8)
        self._counts = _GrowableCounts(self.num_classes, capacity * 8)
        self._tree_lengths = []
        self._leaf_lengths = []
        self._index = []
        self._iters = []
        self._log_weights = []
        # id -> slot for the record being built and the one before it, each
        # with the objects behind those ids kept alive.
        self._curr, self._curr_refs = {}, []
        self._prev, self._prev_refs = {}, []

    # ---------------- recording ---------------- #

    def record(self, states, iteration=None, log_weights=None):
        """Append one iteration's / step's ensemble."""
        if self._index and len(states) != len(self._index[0]):
            # The index is one rectangular array, and both samplers keep their
            # population fixed, so this is a caller bug rather than a case to
            # support with a ragged store.
            raise ValueError(
                f"ensemble size changed: {len(self._index[0])} states were "
                f"recorded per step, this call passed {len(states)}")
        self._prev, self._prev_refs = self._curr, self._curr_refs
        self._curr, self._curr_refs = {}, []

        row = np.empty(len(states), dtype=np.int64)
        for i, state in enumerate(states):
            key = id(state)
            slot = self._curr.get(key)
            if slot is None:
                slot = self._prev.get(key)
                if slot is None:
                    slot = self._store(state)
                self._curr[key] = slot
                self._curr_refs.append(state)
            row[i] = slot
        self._index.append(row)
        self._iters.append(len(self._index) - 1 if iteration is None
                           else int(iteration))
        if log_weights is not None:
            self._log_weights.append(np.asarray(log_weights, dtype=np.float64))

    def _store(self, state):
        """Write one distinct state to the buffers and return its slot."""
        slot = len(self._tree_lengths)

        rows = state.tree
        t_len = len(rows)
        if t_len:
            block = np.asarray(rows, dtype=np.float64)
            self._tree.extend({name: block[:, j]
                               for j, name in enumerate(TREE_COLUMNS)})
        self._tree_lengths.append(t_len)

        counts = state.counts
        l_len = len(counts)
        if l_len:
            ids = np.fromiter(counts.keys(), dtype=np.int64, count=l_len)
            order = np.argsort(ids, kind='stable')
            self._leaf.extend({'ids': ids[order]})
            self._counts.extend(
                np.stack(list(counts.values())).astype(np.int32)[order])
        self._leaf_lengths.append(l_len)
        return slot

    # ---------------- reading back ---------------- #

    def __len__(self):
        return len(self._index)

    @property
    def n_stored(self):
        return len(self._tree_lengths)

    def arrays(self, prefix="state_"):
        """
        The flat arrays, ready to go straight into a results file.

        Prefixed so they can live in the same {column: array} dict as the
        per-iteration columns and the move log without colliding with either.
        """
        out = {prefix + "tree_" + name: col
               for name, col in self._tree.arrays().items()}
        out[prefix + "tree_lengths"] = np.asarray(self._tree_lengths, dtype=np.int32)
        out[prefix + "leaf_lengths"] = np.asarray(self._leaf_lengths, dtype=np.int32)
        out[prefix + "leaf_ids"] = self._leaf.arrays()['ids']
        out[prefix + "leaf_counts"] = self._counts.array()
        out[prefix + "index"] = (np.stack(self._index).astype(np.int32)
                                 if self._index
                                 else np.zeros((0, 0), dtype=np.int32))
        out[prefix + "iters"] = np.asarray(self._iters, dtype=np.int64)
        out[prefix + "num_classes"] = self.num_classes
        out[prefix + "alpha"] = self.alpha
        if self._log_weights:
            out[prefix + "log_weights"] = np.stack(self._log_weights).astype(np.float32)
        return out

    def series(self):
        """A StateSeries over what has been recorded so far."""
        return series_from_arrays(self.arrays())

    def __repr__(self):
        return (f"<StateRecorder {len(self)} records, "
                f"{self.n_stored} distinct states>")


def series_from_arrays(run, prefix="state_"):
    """
    Rebuild a StateSeries from the flat arrays StateRecorder.arrays() produced
    -- whether they are still in memory or have been through a results file.

    None when the run holds no stored states, which is what a run made with
    --no-states looks like.
    """
    key = prefix + "tree_lengths"
    if key not in run:
        return None
    cols = {name: run[prefix + "tree_" + name] for name in TREE_COLUMNS
            if prefix + "tree_" + name in run}
    if not cols:
        # Every stored state is a stump, so no node rows were ever written.
        cols = {name: np.zeros(0, dtype=dt) for name, dt in _TREE_DTYPES}
    return StateSeries(
        cols, run[key], run[prefix + "leaf_lengths"],
        run[prefix + "leaf_ids"], run[prefix + "leaf_counts"],
        run[prefix + "index"], run[prefix + "num_classes"],
        alpha=run.get(prefix + "alpha", 1.0),
        log_weights=run.get(prefix + "log_weights"),
        record_iters=run.get(prefix + "iters"))
