import math

from discretesampling.base.random import RNG
from discretesampling.domain.incremental_decision_tree.problem import BARRED
from discretesampling.domain.incremental_decision_tree.moves import (
    apply_subtree_proposal, change_admissible, draw_subtree,
    evaluate_subtree_move, select_move, subtree_admissible, subtree_node_data)
from discretesampling.domain.incremental_decision_tree.subsampling import (
    assign_blocks, block_subset)
from discretesampling.domain.incremental_decision_tree.proposals.base import (
    IncrementalTreeProposalBase)


class HINTSProposal(IncrementalTreeProposalBase):
    """
    Hierarchical Importance with Nested Training Samples (HINTS)
    as a proposal for the incremental decision tree domain.
    Draw a subtree, assign its rows to blocks, and run one move per block,
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
        # Labels the subtree's rows in place and reports m without ever
        # materialising them.
        num_blocks, block_of, m = assign_blocks(
            x, ctx, subtree_root, rng, self.ss_prop, self.min_data)
        self.n_blocks_total += num_blocks

        state_new, forward, reverse = self._sweep(
            x, ctx, rng, num_blocks, block_of)
        return self._finish_sweep(x, state_new, subtree_root, forward, reverse)

    def _sweep(self, x, ctx, rng, num_blocks, block_of):
        """
        Run a sweep of moves, one per block, applying them to a private copy of
        the tree. Return the new tree and the sum of the surrogate log-densities
        of the accepted moves in both directions.
        The sweep is a single proposal, so the forward and reverse log-densities
        are accumulated and returned to be added to the particle's log proposal
        correction.
        """
        state_new = x
        forward = reverse = 0.0

        for j in range(num_blocks):
            move, node = select_move(state_new, ctx, rng)
            if move == "stay":
                continue

            # Full node data, not the block. The min_samples_leaf test inside
            # evaluate_subtree_move is a property of the proposed tree, and
            # screening it on a fraction of the rows would let HINTS accept
            # trees the other methods reject -- at which point the three are no
            # longer sampling the same target.
            node_data = subtree_node_data(state_new, ctx, node)
            prop_correction, prop_move = evaluate_subtree_move(
                state_new, ctx, move, node, node_data, rng)
            if prop_correction <= BARRED:
                self.n_barred += 1
                continue

            subset = block_subset(node_data, block_of, j)
            v_sub, v_sub_prime, _ = self.target.eval_subset(
                state_new, move, node, prop_move, subset, len(node_data))

            log_accept = min(0.0, v_sub_prime - v_sub + prop_correction)
            if not rng.random() < math.exp(log_accept):
                self.n_screened_out += 1
                continue

            # Every state the sweep passes through has to be admissible, not
            # just the one it ends on. From a state with a starved leaf, a
            # prune can merge that leaf away -- and the grow that would undo it
            # is barred by valid_threshold, so the move has forward probability
            # and no reverse path at all. The correction assumes each inner
            # step is reversible with respect to its own surrogate, so one such
            # move biases the whole sweep. Only a change can starve a leaf.
            if move == "change" and not change_admissible(
                    state_new, node, prop_move['feat'], prop_move['thr'], node_data):
                self.n_inadmissible += 1
                continue

            if state_new is x:
                state_new = x.deep_copy()
            apply_subtree_proposal(state_new, ctx, move, prop_move)
            if self.validate_intermediate and not subtree_admissible(state_new, ctx.root):
                raise AssertionError(
                    f"HINTS sweep left an inadmissible intermediate state after a "
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

        return state_new, forward, reverse

    def __repr__(self):
        return f"<HINTSProposal ss_prop={self.ss_prop} min_data={self.min_data}>"
