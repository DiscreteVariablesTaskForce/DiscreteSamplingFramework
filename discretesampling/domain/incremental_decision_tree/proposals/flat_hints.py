import math

import numpy as np

from discretesampling.base.random import RNG
from discretesampling.domain.incremental_decision_tree.problem import BARRED
from discretesampling.domain.incremental_decision_tree.moves import (
    apply_subtree_proposal, change_partition, draw_subtree,
    evaluate_subtree_move, select_move, subtree_admissible, subtree_leaves,
    subtree_size)
from discretesampling.domain.incremental_decision_tree.subsampling import (
    RowBlocks, block_count, grouped_rows)
from discretesampling.domain.incremental_decision_tree.proposals.base import (
    IncrementalTreeProposalBase)
from discretesampling.domain.incremental_decision_tree.diagnostics import (
    APPLIED, BARRED as BARRED_OUT, INADMISSIBLE, SCREENED_OUT, STAY)


class FlatHINTSProposal(IncrementalTreeProposalBase):
    """
    Flat, single-level Hierarchical Importance with Nested Training Samples
    (HINTS) as a proposal for the incremental decision tree domain.
    HINTSProposal is the nested version.
    Draw a subtree, partition its rows into blocks, and run one move per block,
    applying the moves sequentially to a private copy of the tree.
    The log proposal correction is the sum of the surrogate log-densities of the accepted moves.

    Parameters
    ----------
    target   : IncrementalTreeTarget
        The target distribution to sample from.
    ss_prop  : float, optional
        Approximation of number of blocks, 1/ss_prop. Default 0.1.
    min_data : int, optional
        Minimum number of rows per block. Default None.
    """

    def __init__(self, target, ss_prop=0.1, min_data=None):
        super().__init__()
        self.target = target
        self.problem = target.problem
        self.ss_prop = float(ss_prop)
        self.min_data = int(min_data) if min_data is not None \
            else max(1, self.problem.n_rows // 10)
        self.n_blocks_total = 0
        # Holds each sweep's split of the rows under the subtree root; a new
        # split is drawn into it at the start of every sweep with more than
        # one block.
        self.blocks = RowBlocks(self.problem.n_rows)
        # Assert after every applied inner move that the sweep's working tree
        # is still admissible. Off in production (it is a walk of the subtree
        # per move); the tests turn it on, because an inadmissible intermediate
        # state is the one failure mode of this sweep that shows up only as a
        # few percent of bias in the sampled distribution.
        self.validate_intermediate = False

    def reset_counters(self):
        super().reset_counters()
        self.n_blocks_total = 0

    def counters(self):
        out = super().counters()
        out['blocks'] = self.n_blocks_total
        return out

    def sample(self, x, rng=RNG(), target=None):
        return self._timed(self._sample, x, rng)

    def _sample(self, x, rng):
        subtree_root, ctx = draw_subtree(x, rng)
        num_blocks = block_count(subtree_size(ctx), self.ss_prop, self.min_data)
        self.n_blocks_total += num_blocks

        state_new, forward, reverse = self._sweep(x, ctx, rng, num_blocks)
        return self._finish_sweep(x, state_new, subtree_root, forward, reverse)

    def _sweep(self, x, ctx, rng, num_blocks):
        """
        Run a sweep of moves, one per block, applying them to a private copy of
        the tree. Return the new tree and the sum of the surrogate log-densities
        of the accepted moves in both directions.
        The sweep is a single proposal, so the forward and reverse log-densities
        are accumulated and returned to be added to the particle's log proposal
        correction.
        """
        if num_blocks > 1:
            # A fresh random split of the rows under the subtree root. Because
            # it is new every sweep, the block labels are exchangeable, so the
            # sweep that visits the blocks in reverse order is as likely as
            # this one, and visiting them in label order is enough for the
            # path correction.
            self.blocks.draw(np.concatenate(list(ctx.leaf_idx.values())),
                             num_blocks, rng)

        state_new = x
        forward = reverse = 0.0

        for j in range(num_blocks):
            move, node = select_move(state_new, ctx, rng)
            if move == "stay":
                self._log(move, node, 0, STAY)
                continue

            leaves = subtree_leaves(state_new, ctx, node)
            leaf_rows = [ctx.leaf_idx[leaf] for leaf in leaves]
            n_rows = sum(len(rows) for rows in leaf_rows)
            # The leaf's full rows for a grow, not the block. The
            # min_samples_leaf test inside evaluate_subtree_move is a property
            # of the proposed tree, and screening it on a fraction of the rows
            # would let FlatHINTS accept trees the other methods reject -- at which
            # point the three are no longer sampling the same target.
            prop_correction, prop_move = evaluate_subtree_move(
                state_new, ctx, move, node,
                leaf_rows[0] if move == "grow" else None, rng)
            if prop_correction <= BARRED:
                self.n_barred += 1
                self._log(move, node, n_rows, BARRED_OUT)
                continue

            subset, leaf_pos = (grouped_rows(leaf_rows) if num_blocks == 1
                                else self.blocks.block_rows(leaf_rows, j))
            v_sub, v_sub_prime, _ = self.target.eval_subset(
                state_new, move, node, prop_move, subset, leaf_pos, leaves, n_rows)
            n_sub, dsurr = len(subset), v_sub_prime - v_sub

            log_accept = min(0.0, v_sub_prime - v_sub + prop_correction)
            if not rng.random() < math.exp(log_accept):
                self.n_screened_out += 1
                self._log(move, node, n_rows, SCREENED_OUT, n_sub, dsurr)
                continue

            # Every state the sweep passes through has to be admissible, not
            # just the one it ends on. From a state with a starved leaf, a
            # prune can merge that leaf away -- and the grow that would undo it
            # is barred by valid_split, so the move has forward probability
            # and no reverse path at all. The correction assumes each inner
            # step is reversible with respect to its own surrogate, so one such
            # move biases the whole sweep. Only a change can starve a leaf.
            if move == "change":
                partition = change_partition(
                    state_new, node, prop_move['feat'], prop_move['thr'],
                    np.concatenate(leaf_rows))
                if partition is None:
                    self.n_inadmissible += 1
                    self._log(move, node, n_rows, INADMISSIBLE, n_sub, dsurr)
                    continue
                prop_move['partition'] = partition

            if state_new is x:
                state_new = x.deep_copy()
            apply_subtree_proposal(state_new, ctx, move, prop_move)
            if self.validate_intermediate and not subtree_admissible(state_new, ctx.root):
                raise AssertionError(
                    f"FlatHINTS sweep left an inadmissible intermediate state after a "
                    f"{move} at node {node}: some leaf under {ctx.root} holds fewer "
                    f"than min_samples_leaf rows. Every state in the sweep must be "
                    f"admissible or the inner steps stop being reversible.")

            # Proposal terms cancel and must not be added: for a
            # pi-hat-reversible inner step, Q(y->x)/Q(x->y) is exactly
            # pi-hat(x)/pi-hat(y). The surrogate changes from block to block,
            # so these accumulate rather than telescope -- which is exactly
            # why the total has to be carried on the particle instead of
            # recomputed from (x, x') later.
            forward += v_sub_prime
            reverse += v_sub
            self.n_inner_moves += 1
            self._log(move, node, n_rows, APPLIED, n_sub, dsurr)

        return state_new, forward, reverse

    def __repr__(self):
        return f"<FlatHINTSProposal ss_prop={self.ss_prop} min_data={self.min_data}>"
