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


def block_subset(node_data, block_of, j):
    """
    Return the rows of `node_data` that are assigned to block `j`, using the
    `block_of` buffer produced by assign_blocks. If block_of is None, return
    node_data unchanged.
    """
    if block_of is None or not node_data.size:
        return node_data
    return node_data[block_of[node_data] == j]


def sample_partition_block(m, node_data, rng, ss_prop, min_data):
    """
    Randomly sample a subset of `node_data` to be the rows of block `j` under a
    HINTS proposal. The number of blocks is determined by the number of rows `m'
    under the subtree, the subsample proportion `ss_prop`, and the minimum block
    size `min_data`. The subset is drawn uniformly without replacement.
    """
    n = len(node_data)
    if n == 0 or m <= 0:
        return np.array([], dtype=int)
    num_blocks = block_count(m, ss_prop, min_data)
    if num_blocks == 1:
        return node_data
    h = int(rng.nprng.binomial(n, 1.0 / num_blocks))
    if h == 0:
        return np.array([], dtype=int)
    return rng.nprng.choice(node_data, size=h, replace=False, shuffle=False)
