import math

import numpy as np

from discretesampling.base.random import RNG
from discretesampling.domain.incremental_decision_tree.problem import BARRED, RowReads
from discretesampling.domain.incremental_decision_tree.moves import (
    SubtreeContext, apply_subtree_proposal, change_partition,
    evaluate_subtree_move, make_context, select_move, subtree_leaves,
    subtree_size)
from discretesampling.domain.incremental_decision_tree.subsampling import (
    block_count, draw_block_labels, grouped_rows)
from discretesampling.domain.incremental_decision_tree.proposals.base import (
    IncrementalTreeProposalBase)
from discretesampling.domain.incremental_decision_tree.diagnostics import (
    APPLIED, BARRED as BARRED_OUT, INADMISSIBLE, SCREENED_OUT, STAY)


class BlockRows:
    """
    A leaf's rows in the unit blocks [lo, hi), not yet built.

    A block state holds one of these, in place of an array, for every leaf no
    move has touched since the top of the sweep: those rows are just the leaf's
    rows in the particle, filtered by unit label, so there is nothing to work
    out until something reads them. rows() builds them from the parent node's
    rows -- itself built on demand -- and keeps them, so the copies of a state
    that share a placeholder share the work, and siblings share their parent's.

    The filter keeps the leaf's own row order, which is the order split() leaves
    them in too, so a sweep sees exactly the rows it would have been handed.

    Only valid during the sweep whose labels it reads: the next draw()
    overwrites them. Block states never outlive their sweep -- the particle the
    sweep returns is the original brought forward by replay, with real arrays.
    """
    __slots__ = ('parent', 'lo', 'hi', 'pos', 'labels', 'reads', '_rows',
                 'bucketed', '_order', '_bounds', '_cum')

    def __init__(self, parent, lo, hi, pos, labels, reads, rows=None, bucketed=False):
        self.parent = parent
        self.lo = lo            # minimum unit label in this leaf's block, inclusive
        self.hi = hi            # maximum unit label in this leaf's block, exclusive
        self.pos = pos          # the leaf's row in NestedBlocks' count table
        self.labels = labels
        self.reads = reads      # the RowReads building the rows is charged to
        self._rows = rows
        # bucketed: build any block's rows from the leaf's rows sorted once by
        # unit label (see _from_buckets) instead of filtering every level.
        self.bucketed = bucketed
        self._order = None
        self._bounds = None
        self._cum = None

    def rows(self):
        r = self._rows
        if r is None:
            if self.bucketed:
                r = self._rows = self._from_buckets()
            else:
                p = self.parent.rows()
                lab = self.labels[p]
                self.reads.n += len(p)
                r = self._rows = p[(lab >= self.lo) & (lab < self.hi)]
        return r

    def _from_buckets(self):
        """
        The rows in [lo, hi), from the leaf's rows at the top of the sweep
        sorted once by unit label. A range of units is then a contiguous slice
        of that order, and putting the slice back in position order gives
        exactly the array the level-by-level filter gives, so the sweep is the
        same; only the labels are read once per leaf rather than once per level.
        """
        top = self.parent
        while top.parent is not None:
            top = top.parent
        if top._order is None:
            full = top._rows
            lab = self.labels[full]
            self.reads.n += len(full)
            top._order = np.argsort(lab, kind="stable")
            top._bounds = np.searchsorted(lab[top._order], np.arange(top.hi + 1))
        b = top._bounds
        return top._rows[np.sort(top._order[b[self.lo]:b[self.hi]])]

    def block_counts(self, y, num_classes):
        """
        The leaf's class counts in [lo, hi), for lazy_counts: from a table of
        its counts in every unit, built the first time any of its blocks is
        asked for (one read of each of its rows' class labels, on top of the
        label reads of the buckets) and shared by every block of the sweep.
        """
        top = self.parent if self.parent is not None else self
        while top.parent is not None:
            top = top.parent
        if top._cum is None:
            if top._order is None:
                self._from_buckets() if self is not top else top.child(0, top.hi)._from_buckets()
            full = top._rows
            ys = y[full[top._order]]
            self.reads.n += len(full)
            units = np.repeat(np.arange(top.hi), np.diff(top._bounds))
            table = np.bincount(units * num_classes + ys,
                                minlength=top.hi * num_classes).reshape(top.hi, num_classes)
            cum = np.zeros((top.hi + 1, num_classes), dtype=table.dtype)
            np.cumsum(table, axis=0, out=cum[1:])
            top._cum = cum
        counts = top._cum[self.hi] - top._cum[self.lo]
        counts.flags.writeable = False
        return counts

    def child(self, lo, hi):
        """The same leaf's rows in the narrower range [lo, hi)."""
        return BlockRows(self, lo, hi, self.pos, self.labels, self.reads,
                         bucketed=self.bucketed)


class LazyLeafIdx(dict):
    """
    A block state's leaf_idx: {leaf: rows}, where a leaf's value may still be a
    BlockRows placeholder. Every read hands back the rows, building them if they
    are not built yet, so the tree code never sees a placeholder.

    Copying copies the placeholders, not the rows -- __iter__ is left alone so
    that dict's own copy takes the raw values -- which is what lets a level hand
    each child a state of its own without touching a row. sub() is the same for
    make_context's slice of the mapping.
    """
    __slots__ = ()

    def __getitem__(self, leaf):
        v = dict.__getitem__(self, leaf)
        return v.rows() if type(v) is BlockRows else v

    def get(self, leaf, default=None):
        v = dict.get(self, leaf, default)
        return v.rows() if type(v) is BlockRows else v

    def pop(self, leaf, *default):
        v = dict.pop(self, leaf, *default)
        return v.rows() if type(v) is BlockRows else v

    def values(self):
        return [self[leaf] for leaf in self]

    def items(self):
        return [(leaf, self[leaf]) for leaf in self]

    def copy(self):
        return LazyLeafIdx(self)

    def sub(self, leaves):
        return LazyLeafIdx({leaf: dict.__getitem__(self, leaf)
                            for leaf in leaves if leaf in self})

    def held(self, leaf):
        """The stored value, placeholder or rows, without building anything."""
        return dict.__getitem__(self, leaf)


class LazyCounts(dict):
    """
    A block state's counts for lazy_counts: {leaf: class counts}, where a
    leaf's value may still be the BlockRows of its block, whose counts are only
    worked out if something reads them (a prune adding two leaves' counts, or
    a level's surrogate on the leaves its moves touched). Copies keep the
    placeholders, as LazyLeafIdx's do.
    """
    __slots__ = ('y', 'num_classes')

    def __init__(self, data, y, num_classes):
        super().__init__(data)
        self.y = y
        self.num_classes = num_classes

    def _value(self, v):
        return v.block_counts(self.y, self.num_classes) if type(v) is BlockRows else v

    def __getitem__(self, leaf):
        return self._value(dict.__getitem__(self, leaf))

    def get(self, leaf, default=None):
        return self._value(dict.get(self, leaf, default))

    def pop(self, leaf, *default):
        return self._value(dict.pop(self, leaf, *default))

    def values(self):
        return [self[leaf] for leaf in self]

    def items(self):
        return [(leaf, self[leaf]) for leaf in self]

    def copy(self):
        return LazyCounts(self, self.y, self.num_classes)


class NestedBlocks:
    """
    One sweep's random split of the rows under the subtree root into nested
    blocks, drawn afresh at the start of every sweep.

    draw() splits the rows of I_rho(T) uniformly at random into U unit blocks
    with draw_block_labels: by default it shuffles the rows, cuts them into U
    blocks whose sizes differ by at most one and labels those blocks 0, ...,
    U - 1 in a random order. With equal=False it gives each row an independent,
    uniformly random unit-block label instead, so the blocks are only of
    roughly equal size, at about a third of the cost per draw. A data node is
    a contiguous range of labels, held as (lo, width): its rows are those whose
    label lies in [lo, lo + width). The top of the hierarchy is (0, U), every
    row under the subtree root, and a node's B children are the B equal
    sub-ranges of its own range, so they nest by construction and each is a
    uniform random subset of its parent's rows.

    tabulate() then counts every (leaf, unit, class) once, so any node's counts
    for a leaf, and how many rows the node holds in all, are differences of two
    prefix sums; no level has to touch the rows to know them. The rows
    themselves are only built for the leaves a move actually reads (BlockRows).

    A new split is drawn for every sweep, as HINTS prescribes, so nothing here
    outlives the sweep it was drawn for.

    `reads` is the RowReads the labelling, the table and the rows built from it
    are charged to: the problem's, for a proposal; a private one otherwise.
    """

    def __init__(self, n_rows, y, equal=True, reads=None, bucketed=False, lazy=False,
                 fused=False):
        self.y = y
        self.equal = bool(equal)
        self.fused = bool(fused)
        self.lazy = bool(lazy) or self.fused
        self.bucketed = bool(bucketed) or self.lazy
        self.labels = np.zeros(n_rows, dtype=np.intp)
        self.n_units = 1
        self._cum = None
        self._total = None
        self.reads = RowReads() if reads is None else reads

    def draw(self, rows, n_units, rng):
        """Split `rows` uniformly at random into `n_units` unit blocks."""
        if self.lazy:
            self._draw_with_sizes(rows, n_units, rng)
        else:
            draw_block_labels(self.labels, rows, n_units, rng, self.equal)
        self.reads.n += len(rows)
        self.n_units = n_units

    def _draw_with_sizes(self, rows, n_units, rng):
        """
        draw_block_labels, with the same calls to the generator in the same
        order, also keeping how many rows each unit got: with lazy_counts
        there is no table to read the sizes off, and the shares f need them.
        """
        U = n_units
        if self.equal:
            shuffled = rng.nprng.permutation(rows)
            cut = np.arange(len(rows)) * U // max(len(rows), 1)
            perm = rng.nprng.permutation(U)
            self.labels[shuffled] = perm[cut]
            sizes = np.zeros(U, dtype=np.int64)
            sizes[perm] = np.bincount(cut, minlength=U)
        else:
            drawn = rng.nprng.integers(0, U, size=len(rows))
            self.labels[rows] = drawn
            sizes = np.bincount(drawn, minlength=U)
        self._total = np.concatenate([[0], np.cumsum(sizes)])

    def tabulate(self, leaf_rows, num_classes):
        """
        Class counts of every leaf in every unit block, as prefix sums over the
        units, in one pass over the rows. Returns, for each leaf, the
        placeholder for its rows at the top of the hierarchy.
        """
        U, n_leaves = self.n_units, len(leaf_rows)
        if self.lazy:
            # No table: a leaf's counts are worked out from its buckets only
            # if a level reads them (BlockRows.block_counts).
            held = [BlockRows(None, 0, U, i, self.labels, self.reads, rows=r,
                              bucketed=True)
                    for i, r in enumerate(leaf_rows)]
            if self.fused:
                # fused: each leaf's rows bucketed by unit label as the labels
                # are drawn. draw() hands every row under the subtree root its
                # label, one charged write each; a counting sort that drops the
                # row into its (leaf, unit) bucket in that same pass needs
                # nothing it does not already hold, so nothing more is charged.
                # The buckets are exactly the ones _from_buckets would sort out
                # later (reading every label a second time), so the sweep is
                # the same.
                for h in held:
                    lab = self.labels[h._rows]
                    h._order = np.argsort(lab, kind="stable")
                    h._bounds = np.searchsorted(lab[h._order], np.arange(U + 1))
            return held
        lens = np.fromiter(map(len, leaf_rows), np.intp, n_leaves)
        rows = np.concatenate(leaf_rows)
        # Every row's block label and class label.
        self.reads.n += 2 * len(rows)
        key = np.repeat(np.arange(n_leaves), lens) * U + self.labels[rows]
        table = np.bincount(key * num_classes + self.y[rows],
                            minlength=n_leaves * U * num_classes)
        cum = np.zeros((n_leaves, U + 1, num_classes), dtype=table.dtype)
        np.cumsum(table.reshape(n_leaves, U, num_classes), axis=1, out=cum[:, 1:])
        self._cum = cum
        # Rows in units [0, u), summed over every leaf and class.
        self._total = cum.sum(axis=(0, 2))
        return [BlockRows(None, 0, U, i, self.labels, self.reads, rows=r,
                          bucketed=self.bucketed)
                for i, r in enumerate(leaf_rows)]

    def counts(self, pos, lo, hi):
        """(len(pos), K) class counts of the tabulated leaves `pos` in units
        [lo, hi). Read-only, like split()'s."""
        counts = self._cum[pos, hi] - self._cum[pos, lo]
        counts.flags.writeable = False
        return counts

    def share(self, lo, hi):
        """
        Fraction of the rows under the subtree root that fell into units
        [lo, hi). With i.i.d. labels this is only (hi - lo) / U on average.
        """
        return float(self._total[hi] - self._total[lo]) / float(self._total[-1])

    def ranges(self, lo, width, branching):
        """The B child ranges of (lo, width), as (lo, width) pairs."""
        child = width // branching
        return [(lo + j * child, child) for j in range(branching)]

    def order(self, branching, rng):
        """
        A uniform permutation to visit a node's children in: the path
        correction needs a sweep and its reverse to be equally likely.
        """
        return rng.nprng.permutation(branching)

    def split(self, rows, lo, width, branching, num_classes):
        """
        One leaf's rows and class counts in each of the B children of the data
        node (lo, width), as [(rows, counts)] in range order. `rows` must lie in
        that node, which holds for every leaf of a state restricted to it.

        Only the leaves a move has rewritten mid-sweep come through here; every
        other leaf is read off the table and left as a placeholder.
        """
        # Every row's block label and class label.
        self.reads.n += 2 * len(rows)
        child = (self.labels[rows] - lo) // (width // branching)
        sizes = np.bincount(child, minlength=branching)
        counts = np.bincount(child * num_classes + self.y[rows],
                             minlength=branching * num_classes)
        counts = counts.reshape(branching, num_classes)
        counts.flags.writeable = False
        # A uint8 key lets numpy's stable sort run as a radix sort, in one pass.
        key = child.astype(np.uint8) if branching < 256 else child
        pieces = np.split(rows[np.argsort(key, kind="stable")],
                          np.cumsum(sizes)[:-1])
        return list(zip(pieces, counts))


class HINTSProposal(IncrementalTreeProposalBase):
    """
    Hierarchical HINTS for the incremental decision tree domain: the nested
    version of FlatHINTSProposal, over L levels of nested blocks rather than one
    flat sweep of B blocks.

    Parameters
    ----------
    target    : IncrementalTreeTarget
        The target distribution to sample from.
    ss_prop   : float, optional
        Approximate fraction of the rows in a primitive block, so the total
        number of primitive blocks is about 1/ss_prop. Default 0.1.
    min_data  : int, optional
        Minimum number of rows per primitive block. Default n_rows // 10.
    levels    : int, optional
        Number of data levels below the top. 2 means top -> B blocks -> B*B
        blocks, with the moves proposed in the B*B primitive blocks. Reduced
        automatically when the subtree has too few rows to split that far.
        Default 2.
    branching : int, optional
        Children per data node. Default: whatever makes the primitive blocks
        come out at about `ss_prop` of the rows.
    hierarchy : sequence of int, optional
        The whole hierarchy in Chapter 2's notation [m_0, m_1, ..., m_H]: m_0
        blocks of ss_prop of the rows at each unit vertex, and branch factor
        m_h at level h, counted up from the unit vertices, so [1,2,4] is four
        vertices under the top, each with two unit vertices of one block, and
        [4,2] is two unit vertices of four blocks each. Replaces `levels` and
        `branching`, which must then be left unset, and ss_prop must be
        1 / (m_0 m_1 ... m_H). On a subtree too small for every unit vertex to
        hold min_data rows, the unit vertices take in the level below them
        (m_1 is dropped first, then m_2, ...) until they can.
    bucket_rows : bool, optional
        Build a leaf's rows in a block from its rows sorted once by unit label
        (True) rather than by filtering the labels again at every level (False,
        the default). The sweep is exactly the same; only fewer labels are read.
    lazy_counts : bool, optional
        Work out a leaf's class counts in a block only when a level reads
        them, and give each level its surrogate's change over the leaves its
        moves touched, rather than tabulating every leaf in every unit and
        summing every leaf at every level (implies bucket_rows). Leaves no
        move touches add the same to a level's surrogate before and after, so
        the acceptance probabilities are the same; only the floating-point
        rounding of the sums differs.
    fused_buckets : bool, optional
        With lazy_counts (which it implies), bucket each leaf's rows by unit
        label in the same pass that draws the labels, rather than reading every
        label back later to sort them. The sweep is exactly the same; only the
        second read of each label is saved.
    temper : bool, optional
        Temper each vertex's surrogate by its share f of the rows (True, the
        default), or use the scaled surrogate at full strength (False). Both
        keep the sampler exact.
    equal_blocks : bool, optional
        Cut the rows into unit blocks of equal size, labelled in a random order
        (True), or give every row an independent, uniformly random unit-block
        label (False), which costs about a third as much per draw but gives
        blocks of binomial size. Default True.

    The rows under the subtree root are split into blocks afresh at the start
    of every sweep (NestedBlocks), as HINTS prescribes.
    """

    def __init__(self, target, ss_prop=0.1, min_data=None, levels=None,
                 branching=None, equal_blocks=True, hierarchy=None, temper=True,
                 bucket_rows=False, lazy_counts=False, fused_buckets=False):
        super().__init__()
        self.target = target
        self.problem = target.problem
        self.ss_prop = float(ss_prop)
        self.min_data = int(min_data) if min_data is not None \
            else max(1, self.problem.n_rows // 10)
        # Each vertex's surrogate is its share f of the rows' scaled log
        # posterior times f (tempered, the default), or the scaled log
        # posterior itself (temper=False, as in Chapter 2). Either is a valid
        # surrogate: the path correction makes the sweep exact for any.
        self.temper = bool(temper)
        self.hierarchy = None if hierarchy is None else tuple(int(m) for m in hierarchy)
        if self.hierarchy is not None:
            if levels is not None or branching is not None:
                raise ValueError("give either hierarchy or levels/branching, not both")
            if len(self.hierarchy) < 2 or self.hierarchy[0] < 1 \
                    or any(m < 2 for m in self.hierarchy[1:]):
                raise ValueError(f"hierarchy must be [m_0 >= 1, m_1 >= 2, ...], "
                                 f"got {list(hierarchy)}")
            blocks = math.prod(self.hierarchy)
            if abs(self.ss_prop * blocks - 1.0) > 1e-9:
                raise ValueError(f"hierarchy {list(self.hierarchy)} has {blocks} "
                                 f"blocks, so ss_prop must be 1/{blocks}, got {ss_prop}")
            levels = len(self.hierarchy) - 1
        self.levels = 2 if levels is None else int(levels)
        self.branching = None if branching is None else int(branching)
        # Refused rather than clamped: a value quietly moved to the nearest
        # legal one is a run that reports settings it did not use.
        if self.levels < 0:
            raise ValueError(f"levels must be >= 0, got {levels}")
        if self.branching is not None and self.branching < 2:
            raise ValueError(f"branching must be >= 2 (or None to derive it), "
                             f"got {branching}")

        # Holds each sweep's split of the rows under the subtree root; a new
        # split is drawn into it at the start of every sweep that has more
        # than one block.
        self.lazy_counts = bool(lazy_counts) or bool(fused_buckets)
        self.blocks = NestedBlocks(self.problem.n_rows, self.problem.y,
                                   equal=equal_blocks, reads=self.problem.reads,
                                   bucketed=bucket_rows, lazy=self.lazy_counts,
                                   fused=fused_buckets)
        self._last_delta = 0.0
        self.n_blocks_total = 0
        self.n_level_accepts = 0
        self.n_level_rejects = 0
        # This sweep's shape, fixed in _sample before the recursion starts: the
        # branching factor has to be the same on the reverse path, so it cannot
        # be read off a state the sweep is changing. _factors holds the branch
        # factor of each level, the top level's first; _levels and _branching
        # describe it when every level has the same factor (else _branching 0).
        self._factors = ()
        self._levels = 0
        self._branching = 0
        # Assert after every applied primitive move that every leaf under the
        # subtree root still meets min_samples_leaf on the full rows, and that
        # the full map matches the tree. Off in production; the tests turn it on.
        self.validate_intermediate = False

    # ------------------- instrumentation ------------------- #

    def reset_counters(self):
        super().reset_counters()
        self.n_blocks_total = 0
        self.n_level_accepts = 0
        self.n_level_rejects = 0

    def counters(self):
        out = super().counters()
        out['blocks'] = self.n_blocks_total
        out['level_accepts'] = self.n_level_accepts
        out['level_rejects'] = self.n_level_rejects
        return out

    # ------------------- the sweep ------------------- #

    def sample(self, x, rng=RNG(), target=None):
        return self._timed(self._sample, x, rng)

    def _sample(self, x, rng):
        subtree_root, ctx = self._draw_root(x, rng)
        total = block_count(subtree_size(ctx), self.ss_prop, self.min_data)
        self._factors = self._sweep_factors(total)
        self._levels = len(self._factors)
        self._branching = (self._factors[0] if len(set(self._factors)) == 1 else 0)
        n_units = math.prod(self._factors)
        self.n_blocks_total += n_units

        if not self._factors:
            # Too few rows to split: the top level is primitive, at f = 1, where
            # the surrogate is the target, so the outer accept only sees the
            # root-draw term.
            state_new = x.deep_copy()
            new_ctx = make_context(state_new, ctx.root)
            pi_in = self._log_pi(state_new, ctx.root, 1.0)
            applied = self._primitive(state_new, new_ctx, dict(new_ctx.leaf_idx),
                                      rng, 1.0)
            if not applied:
                return self._finish_sweep(x, x, subtree_root, 0.0, 0.0)
            pi_out = self._log_pi(state_new, ctx.root, 1.0)
            return self._finish_sweep(x, state_new, subtree_root, pi_out, pi_in)

        # A fresh random split of the rows under the subtree root, into one
        # unit block per unit vertex.
        self.blocks.draw(np.concatenate(list(ctx.leaf_idx.values())), n_units, rng)

        forward, reverse, applied = self._children(
            x, ctx, dict(ctx.leaf_idx), rng, 0, n_units, 0)
        if not applied:
            return self._finish_sweep(x, x, subtree_root, 0.0, 0.0)
        # The top level holds every row, so this is where the sweep's moves are
        # finally paid for on all of them.
        state_new = self._replayed(x, ctx.root, applied)
        return self._finish_sweep(x, state_new, subtree_root, forward, reverse)

    def _sweep_factors(self, total):
        """
        The branch factor of each level of this sweep's hierarchy, the top
        level's first, for a subtree that splits into about `total` blocks of
        at least min_data rows; () for a sweep too small to split at all.
        """
        if self.hierarchy is None:
            levels, branching = self._shape(total)
            return (branching,) * levels
        factors = list(self.hierarchy[1:])  # the unit vertices' parents first
        while factors and math.prod(factors) > total:
            # Too many unit vertices for their rows: each unit vertex takes in
            # the level below it, so it is a level that goes, never a block
            # that starves.
            factors.pop(0)
        return tuple(reversed(factors))

    def _shape(self, total):
        """
        (levels, branching) for a sweep whose subtree splits into about `total`
        primitive blocks. Levels are dropped rather than blocks when the subtree
        is too small to feed them: a level with one child adds an accept step
        that can only agree with the child below it, and charges a pass over
        its own rows for the privilege.

        With a fixed branching factor the test is against that factor, not 2:
        otherwise a wide hierarchy (8 x 8, say) would keep all of its unit
        blocks on a subtree that can only feed a few, and its blocks would hold
        fewer than min_data rows, some of them none.
        """
        levels = self.levels
        smallest = self.branching if self.branching is not None else 2
        while levels > 0 and smallest ** levels > total:
            levels -= 1
        if levels == 0:
            return 0, 0
        if self.branching is not None:
            return levels, self.branching
        return levels, max(2, int(total ** (1.0 / levels)))

    def _node(self, state, ctx, full, rng, lo, width, depth):
        """
        Run one data node's kernel on `state`, which already holds that node's
        rows and no others; `full` is the full rows of its leaves under the
        subtree root.

        Returns (pi_in, pi_out, applied): this node's own surrogate at the
        states it started and ended on -- the two halves of the term its parent
        adds to the path correction -- and the moves it settled on, as the
        (move, prop_move, full_after) triples any state of the same topology
        can be brought forward by. A node that rejects returns no moves and two
        equal densities, so it contributes nothing to its parent's correction.
        """
        f = self.blocks.share(lo, lo + width)
        if f == 0.0:
            return 0.0, 0.0, ()
        if self.lazy_counts:
            return self._node_lazy(state, ctx, full, rng, lo, width, depth, f)
        pi_in = self._log_pi(state, ctx.root, f)

        if depth >= len(self._factors):
            applied = self._primitive(state, ctx, full, rng, f)
            if not applied:
                return pi_in, pi_in, ()
            return pi_in, self._log_pi(state, ctx.root, f), applied

        forward, reverse, applied = self._children(state, ctx, full, rng,
                                                   lo, width, depth)
        if not applied:
            return pi_in, pi_in, ()

        new = self._replayed(state, ctx.root, applied)
        pi_out = self._log_pi(new, ctx.root, f)
        # The HINTS path correction: what this node's own rows make of the
        # composite move, less what each child already charged for its part of
        # it. The children's terms are what stops a move that only its own
        # block liked from being waved through here.
        log_accept = min(0.0, (pi_out - pi_in) - (forward - reverse))
        if rng.random() < math.exp(log_accept):
            self.n_level_accepts += 1
            return pi_in, pi_out, applied
        self.n_level_rejects += 1
        return pi_in, pi_in, ()

    def _node_lazy(self, state, ctx, full, rng, lo, width, depth, f):
        """
        _node for lazy_counts. Only the change in a level's surrogate is ever
        used (by its own accept step and, through the children's terms, by the
        level above), so a node reports (0, change) for what it did: the unit
        vertex the change its screen already worked out, and a level the
        change over the leaves below the nodes its moves touched, every other
        leaf adding the same before and after.
        """
        if depth >= len(self._factors):
            applied = self._primitive(state, ctx, full, rng, f)
            if not applied:
                return 0.0, 0.0, ()
            return 0.0, self._last_delta, applied

        forward, reverse, applied = self._children(state, ctx, full, rng,
                                                   lo, width, depth)
        if not applied:
            return 0.0, 0.0, ()

        new = self._replayed(state, ctx.root, applied)
        change = self._surrogate_change(state, new, ctx.root, applied, f)
        log_accept = min(0.0, change - (forward - reverse))
        if rng.random() < math.exp(log_accept):
            self.n_level_accepts += 1
            return 0.0, change, applied
        self.n_level_rejects += 1
        return 0.0, 0.0, ()

    def _surrogate_change(self, old, new, root, applied, f):
        """
        _log_pi(new) - _log_pi(old), summing the leaf likelihoods only over
        the leaves below the nodes the moves were made at: the rest are the
        same leaves with the same rows in both, so they cancel exactly.
        """
        moved = {int(prop['node']) for _, prop, _ in applied}

        def touched(state):
            out = set()
            for node in moved:
                if node in state.nodes or node in state.leaf_idx:
                    out.update(state._descendant_leaves(node))
            return sorted(out)

        target = self.target
        value = 0.0
        for state, sign in ((new, 1.0), (old, -1.0)):
            leaves = touched(state)
            value += sign * self._prior_pi(state, root)
            if leaves:
                counts = np.stack([state.counts[leaf] for leaf in leaves])
                value += sign * target._sum_leaf_log_dm(counts, range(len(leaves)), 1.0 / f)
        return (f if self.temper else 1.0) * value

    def _prior_pi(self, state, root):
        """The prior part of _log_pi: no rows read."""
        problem = self.problem
        m = state.nodes
        lp_vals = problem.lp_vals
        stack, n_internal, splits = [root], 0, 0.0
        while stack:
            node = stack.pop()
            row = m.get(node)
            if row is None:
                continue
            n_internal += 1
            splits += lp_vals[int(row[3])]
            stack.append(int(row[1]))
            stack.append(int(row[2]))
        n_nodes = len(state.tree)
        return (self.target.log_prior(n_nodes, n_nodes + 1)
                + splits + n_internal * problem.lp_feats)

    def _children(self, state, ctx, full, rng, lo, width, depth):
        """
        Run this node's B children in a random order, each starting from where
        the last one stopped. Returns the two halves of the path correction and
        the moves the children settled on, in the order they were applied.

        Every child starts from `state` -- this node's entry state, which does
        not change here -- restricted to its own block, and is then brought
        forward by replaying whatever the earlier children did, along with the
        full map the last of those moves left. A child's rows are only ever
        touched by the moves that are actually applied, so the hand-off from
        one sibling to the next costs the moves, not the node's rows.

        Nor is a leaf's share of the rows split out here. A leaf no move has
        touched keeps its rows in the particle, so its counts in each child come
        off the table the top level built, and its rows are a placeholder that
        is only filled in if a move reads it. Only leaves a move has rewritten
        mid-sweep hold real arrays, and only those are split.
        """
        B, K = self._factors[depth], self.problem.num_classes
        ranges = self.blocks.ranges(lo, width, B)
        order = self.blocks.order(B, rng)
        base_leaves = state._descendant_leaves(ctx.root)
        if depth == 0:
            held = self.blocks.tabulate(
                [state.leaf_idx[leaf] for leaf in base_leaves], K)
        else:
            held = [state.leaf_idx.held(leaf) for leaf in base_leaves]

        untouched = [i for i, h in enumerate(held) if type(h) is BlockRows]
        pos = np.fromiter((held[i].pos for i in untouched), np.intp, len(untouched))
        rewritten = {i: self.blocks.split(h, lo, width, B, K)
                     for i, h in enumerate(held) if type(h) is not BlockRows}
        pieces = []
        lazy = self.lazy_counts
        for j, (child_lo, child_width) in enumerate(ranges):
            child_hi = child_lo + child_width
            piece = [None] * len(held)
            if lazy:
                # The block's rows stand in for its counts too (LazyCounts).
                for i in untouched:
                    c = held[i].child(child_lo, child_hi)
                    piece[i] = (c, c)
            else:
                counts = self.blocks.counts(pos, child_lo, child_hi)
                for k, i in enumerate(untouched):
                    piece[i] = (held[i].child(child_lo, child_hi), counts[k])
            for i, parts in rewritten.items():
                piece[i] = parts[j]
            pieces.append(piece)

        applied, forward, reverse = [], 0.0, 0.0
        for j in order:
            child_lo, child_width = ranges[j]
            child = self._restricted(state, base_leaves, pieces[j])
            child_ctx = make_context(child, ctx.root)
            if applied:
                self._replay(child, child_ctx, applied)
            child_full = applied[-1][2] if applied else full
            pi_in, pi_out, child_applied = self._node(
                child, child_ctx, child_full, rng, child_lo, child_width,
                depth + 1)
            # Proposal terms cancel and must not be added: a child's kernel is
            # reversible for its own surrogate, so Q(y->x)/Q(x->y) is exactly
            # pi_c(x)/pi_c(y). The surrogate differs from child to child, so
            # these accumulate rather than telescope.
            forward += pi_out
            reverse += pi_in
            applied.extend(child_applied)
        return forward, reverse, applied

    def _primitive(self, state, ctx, full, rng, f):
        """
        One Metropolis-Hastings step against this block's surrogate, applied in
        place to `state` -- which is this proposal's own copy and holds only
        this block's rows. Admissibility is judged on `full`, the surrogate on
        the block's rows. Returns [(move, prop_move, full_after)] if the move
        was kept, else [].

        The accepted delta is built from the move here and the density it
        belongs to is summed over the whole subtree in _log_pi. They have to be
        the same function of the state, or the level above subtracts something
        this one never charged.
        """
        # The same tree, seen through the full rows: select_move's size test
        # for a grow or change reads the row counts out of the context.
        full_ctx = SubtreeContext(ctx.root, ctx.nodes, ctx.terminal_nodes,
                                  ctx.leaves, full)
        move, node = select_move(state, full_ctx, rng)
        if move == "stay":
            self._log(move, node, 0, STAY)
            return []

        leaves = subtree_leaves(state, ctx, node)
        full_rows = [full[leaf] for leaf in leaves]
        n_rows = sum(len(rows) for rows in full_rows)
        # The leaf's full rows for a grow, so min_samples_leaf is tested on the
        # proposed tree rather than on this block's share of it.
        prop_correction, prop_move = evaluate_subtree_move(
            state, ctx, move, node, full_rows[0] if move == "grow" else None, rng)
        if prop_correction <= BARRED:
            self.n_barred += 1
            self._log(move, node, n_rows, BARRED_OUT)
            return []

        subset, leaf_pos = grouped_rows([ctx.leaf_idx[leaf] for leaf in leaves])
        n_sub = len(subset)
        if n_sub == 0:
            # No row of this block reaches the node: nothing to judge it on.
            self.n_screened_out += 1
            self._log(move, node, n_rows, SCREENED_OUT, 0)
            return []

        # eval_move returns the scaled surrogate's two values, prior included;
        # tempering is the factor f on their difference.
        v, v_prime = self.target.eval_move(state, move, node, prop_move,
                                           subset, leaf_pos, leaves, 1.0 / f)
        delta = (f if self.temper else 1.0) * (v_prime - v)
        log_accept = min(0.0, delta + prop_correction)
        if not rng.random() < math.exp(log_accept):
            self.n_screened_out += 1
            self._log(move, node, n_rows, SCREENED_OUT, n_sub, delta)
            return []

        # A grow's split was tested in evaluate_subtree_move and a prune of an
        # admissible tree is admissible, so only a change is left to check.
        if move == "change":
            partition = change_partition(state, node, prop_move['feat'],
                                         prop_move['thr'], np.concatenate(full_rows))
            if partition is None:
                self.n_inadmissible += 1
                self._log(move, node, n_rows, INADMISSIBLE, n_sub, delta)
                return []

        full_after = dict(full)
        if move == "prune":
            L, R = state._child_nodes(node)
            full_after[node] = np.concatenate([full_after.pop(L), full_after.pop(R)])
        elif move == "change":
            full_after.update(partition)

        # The plain move: the split mask and partition above are over the full
        # rows, and this state holds only the block's. A level replays it onto
        # other rows, which have to be split and routed themselves.
        plain = {'node': node, 'feat': prop_move['feat'], 'thr': prop_move['thr']}
        apply_subtree_proposal(state, ctx, move, plain)

        if move == "grow":
            rows = full_after.pop(node)
            go_left = prop_move.get('go_left')
            if go_left is None:
                go_left = self.problem.X[rows, int(prop_move['feat'])] < prop_move['thr']
                self.problem.reads.n += len(rows)
            L, R = state._child_nodes(node)
            full_after[L] = rows[go_left]
            full_after[R] = rows[~go_left]

        if self.validate_intermediate:
            self._check_full(state, ctx.root, full_after, move, node)

        self.n_inner_moves += 1
        self._log(move, node, n_rows, APPLIED, n_sub, delta)
        self._last_delta = delta
        return [(move, plain, full_after)]

    def _check_full(self, state, root, full, move, node):
        leaves = state._descendant_leaves(root)
        if set(leaves) != set(full):
            raise AssertionError(
                f"HINTS's full map after a {move} at node {node} holds leaves "
                f"{sorted(full)}, but the tree under {root} has {sorted(leaves)}.")
        min_leaf = self.problem.min_samples_leaf
        starved = [leaf for leaf in leaves if len(full[leaf]) < min_leaf]
        if starved:
            raise AssertionError(
                f"HINTS sweep left an inadmissible intermediate state after a "
                f"{move} at node {node}: leaves {starved} under {root} hold fewer "
                f"than min_samples_leaf rows.")

    # ------------------- states restricted to a data node ------------------- #

    def _restricted(self, base, base_leaves, pieces):
        """
        A private copy of `base` holding one block's rows: `pieces[i]` is the
        (rows, counts) of `base_leaves[i]` in that block, the rows either an
        array or a BlockRows placeholder.

        Restriction commutes with every move -- a row's leaf is decided by the
        splits above it and not by which block it is in -- so the leaves come
        straight from `base`'s own leaves, sliced. No row is routed here.
        """
        new = base.deep_copy()
        leaf_idx = new.leaf_idx
        if type(leaf_idx) is not LazyLeafIdx:
            leaf_idx = new.leaf_idx = LazyLeafIdx(leaf_idx)
        if self.lazy_counts and type(new.counts) is not LazyCounts:
            new.counts = LazyCounts(new.counts, self.problem.y, self.problem.num_classes)
        counts_map = new.counts
        for leaf, (idx, counts) in zip(base_leaves, pieces):
            dict.__setitem__(leaf_idx, leaf, idx)
            dict.__setitem__(counts_map, leaf, counts)
        return new

    def _replay(self, state, ctx, applied):
        """
        Bring `state` forward by moves settled elsewhere in the sweep, in the
        order they were applied.

        The moves land on the same nodes and make the same new ones: node ids
        are handed out in order from a counter `state` shares with the state
        the moves were drawn on, and the sequence replayed here is the sequence
        that was applied there.
        """
        for move, prop_move, _ in applied:
            apply_subtree_proposal(state, ctx, move, prop_move)

    def _replayed(self, base, root, applied):
        """
        A copy of `base` brought forward by `applied` -- the composite move a
        level's children built, paid for on the rows of the level above them.

        At the top level `base` is the particle itself, so this is the copy the
        sampler gets back.
        """
        new = base.deep_copy()
        self._replay(new, make_context(new, root), applied)
        return new

    # ------------------- the tempered surrogate ------------------- #

    def _log_pi(self, state, root, f):
        """
        This data node's surrogate log-density of `state`, up to a constant that
        is the same for every state the node sees -- the leaves outside the
        subtree, which no move touches. Only differences within one level are
        ever taken, so the constant cancels wherever this is used.
        """
        target, problem = self.target, self.problem
        m = state.nodes
        lp_vals = problem.lp_vals
        stack, counts, n_internal, splits = [root], [], 0, 0.0
        while stack:
            node = stack.pop()
            row = m.get(node)
            if row is None:
                counts.append(state.counts[node])
                continue
            n_internal += 1
            splits += lp_vals[int(row[3])]
            stack.append(int(row[1]))
            stack.append(int(row[2]))

        # A binary tree's leaves are its internal nodes plus one, so the whole
        # tree's shape prior is read off the node count without touching
        # leaf_idx -- which holds this node's restricted rows, not the tree's.
        n_nodes = len(state.tree)
        value = (target.log_prior(n_nodes, n_nodes + 1)
                 + splits + n_internal * problem.lp_feats)
        if counts:
            value += target._sum_leaf_log_dm(np.stack(counts),
                                             range(len(counts)), 1.0 / f)
        return (f if self.temper else 1.0) * value

    def __repr__(self):
        shape = (f"hierarchy={list(self.hierarchy)}" if self.hierarchy is not None
                 else f"levels={self.levels}")
        return (f"<HINTSProposal ss_prop={self.ss_prop} min_data={self.min_data} "
                f"{shape}>")
