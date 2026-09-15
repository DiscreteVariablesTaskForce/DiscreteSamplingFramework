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


def assign_blocks(state, ctx, subtree_root, rng, ss_prop, min_data):
    """
    Assign the rows under `subtree_root` to blocks, for a HINTS proposal.
    Returns (num_blocks, buf, m), where:
    - num_blocks is the number of blocks to partition the rows into,
    - buf is a 1D array of length n_rows, filled with -1 except for the rows under
      `subtree_root`, which are labeled with their block number,
    - m is the number of rows under `subtree_root`.
    """
    leaf_arrays = [ctx.leaf_idx[leaf]
                   for leaf in state._descendant_leaves(subtree_root)
                   if leaf in ctx.leaf_idx]
    m = sum(len(a) for a in leaf_arrays)
    num_blocks = block_count(m, ss_prop, min_data)
    if num_blocks == 1:
        return 1, None, m
    buf = state.problem.block_buffer()
    draw = rng.nprng.integers
    for a in leaf_arrays:
        buf[a] = draw(0, num_blocks, size=len(a), dtype=np.int16)
    return num_blocks, buf, m


def block_rows(ctx, leaves, block_of, j, labels):
    """
    Block j's rows under the node whose leaves are `leaves`, as
    (subset, leaf_pos) -- see grouped_rows -- using the `block_of` buffer
    produced by assign_blocks; every row under the node if block_of is None.

    `labels` is a dict held for one sweep. Each leaf's block labels are read
    out of block_of once and reused for every later block, until a move
    replaces that leaf's rows.
    """
    leaf_rows = [ctx.leaf_idx[leaf] for leaf in leaves]
    if block_of is None:
        return grouped_rows(leaf_rows)
    picked = []
    for leaf, rows in zip(leaves, leaf_rows):
        cached = labels.get(leaf)
        if cached is None or cached[0] is not rows:
            cached = labels[leaf] = (rows, block_of[rows])
        picked.append(rows[cached[1] == j])
    return grouped_rows(picked)


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
