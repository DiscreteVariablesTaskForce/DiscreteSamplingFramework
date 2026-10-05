import numpy as np


def block_count(m, ss_prop, min_data):
    """
    Given the number of rows m under a subtree, return the number of blocks to
    partition them into for a HINTS proposal. The block count is the number of
    blocks that would be drawn if the subtree's rows were assigned to blocks
    i.i.d. uniform, with the given subsample proportion and minimum block size.
    """
    subset_size = max(int(min_data), int(m * ss_prop), 1)
    return max(1, m // subset_size)


def grouped_rows(leaf_rows):
    """
    The rows of several leaves as one array, and for each row the position in
    `leaf_rows` of the leaf it came from -- the form
    IncrementalTreeTarget.eval_subset takes a subset in.
    """
    rows = np.concatenate(leaf_rows) if len(leaf_rows) != 1 else leaf_rows[0]
    return rows, np.repeat(np.arange(len(leaf_rows)), [len(a) for a in leaf_rows])


class RowBlocks:
    """
    One sweep's random split of the rows under the subtree root into blocks,
    for a FlatHINTS sweep, drawn afresh at the start of every sweep.

    draw() gives each row under the subtree root an independent, uniformly
    random block label in 0, ..., B - 1, so every block is a uniform random
    subset of those rows and the blocks are of roughly equal size. Nothing
    outlives the sweep it was drawn for.
    """

    def __init__(self, n_rows):
        self.labels = np.zeros(n_rows, dtype=np.intp)

    def draw(self, rows, num_blocks, rng):
        """Split `rows` uniformly at random into `num_blocks` blocks."""
        self.labels[rows] = rng.nprng.integers(0, num_blocks, size=len(rows))

    def block_rows(self, leaf_rows, j):
        """
        The rows of block `j` in each of `leaf_rows`, as (subset, leaf_pos):
        see grouped_rows. Every row of `leaf_rows` must be one the current
        split was drawn over.
        """
        return grouped_rows([rows[self.labels[rows] == j] for rows in leaf_rows])


def sample_partition_block(m, leaf_rows, rng, ss_prop, min_data):
    """
    Randomly sample the rows of one block of a HINTS partition of the rows
    under a node, given as the row arrays of the leaves under it. The number of
    blocks is determined by the number of rows `m` under the subtree, the
    subsample proportion `ss_prop`, and the minimum block size `min_data`. The
    subset is drawn uniformly without replacement, and returned as
    (subset, leaf_pos) -- see grouped_rows.
    """
    lengths = [len(a) for a in leaf_rows]
    n = sum(lengths)
    if n == 0 or m <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    num_blocks = block_count(m, ss_prop, min_data)
    if num_blocks == 1:
        return grouped_rows(leaf_rows)
    h = int(rng.nprng.binomial(n, 1.0 / num_blocks))
    if h == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    # Positions rather than rows: the same rows are drawn, and each position
    # also says which leaf its row came from.
    idx = rng.nprng.choice(n, size=h, replace=False, shuffle=False)
    rows = np.concatenate(leaf_rows) if len(leaf_rows) != 1 else leaf_rows[0]
    return rows[idx], np.searchsorted(np.cumsum(lengths), idx, side='right')
