import math

from discretesampling.base.random import RNG
from discretesampling.domain.incremental_decision_tree.problem import BARRED
from discretesampling.domain.incremental_decision_tree.moves import (
    draw_subtree, evaluate_subtree_move, select_move, subtree_node_data,
    subtree_size)
from discretesampling.domain.incremental_decision_tree.subsampling import (
    block_count, sample_partition_block)
from discretesampling.domain.incremental_decision_tree.proposals.base import (
    IncrementalTreeProposalBase)


class DAProposal(IncrementalTreeProposalBase):
    """
    Delayed Acceptance proposal for the incremental decision tree domain.
    Draw a subtree, draw a move inside it, apply it, and return the new tree
    and the log proposal correction.
    The move is screened using a surrogate target that is cheaper to evaluate than the full target.
    If the move is rejected by the surrogate, the original tree is returned without evaluating the full target.

    Parameters
    ----------
    target   : IncrementalTreeTarget
        The target distribution to sample from.
    ss_prop  : float, optional
        Proportion of Data to subsample for surrogate. Default 0.1.
    min_data : int, optional
        Minimum Data to subsample for surrogate. Default None.
    """

    def __init__(self, target, ss_prop=0.1, min_data=None):
        super().__init__()
        self.target = target
        self.problem = target.problem
        self.ss_prop = float(ss_prop)
        self.min_data = int(min_data) if min_data is not None \
            else max(1, self.problem.n_rows // 10)

    def sample(self, x, rng=RNG(), target=None):
        return self._timed(self._sample, x, rng)

    def _sample(self, x, rng):
        subtree_root, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            self._note(move, node, 0)
            return self._stay(x)

        node_data = subtree_node_data(x, ctx, node)
        self._note(move, node, len(node_data))
        prop_correction, prop_move = evaluate_subtree_move(
            x, ctx, move, node, node_data, rng)
        if prop_correction <= BARRED:
            # Dead already: the screen has nothing to screen.
            return self._stay(x, 'n_barred')

        m = subtree_size(ctx)
        if (len(node_data) < self.min_data
                and block_count(m, self.ss_prop, self.min_data) > 1):
            # Too few rows reach this node for a block of it to say anything.
            # Skip the screen, exactly as MCMC_DA's da_flag does, and let the
            # weight update carry the whole move: with no screen the kernel is
            # just q, so the correction is the plain proposal ratio.
            self.n_inner_moves += 1
            return self._finish(x, ctx, move, prop_move, subtree_root,
                                0.0, prop_correction)

        # One block, simulated rather than built: only its overlap with this
        # node's rows is ever needed, and that is O(overlap) to draw.
        subset = sample_partition_block(m, node_data, rng,
                                        self.ss_prop, self.min_data)
        v_sub, v_sub_prime, _ = self.target.eval_subset(
            x, move, node, prop_move, subset, len(node_data))
        self._note(move, node, len(node_data), len(subset), v_sub_prime - v_sub)

        log_accept = min(0.0, v_sub_prime - v_sub + prop_correction)
        if not rng.random() < math.exp(log_accept):
            return self._stay(x, 'n_screened_out')

        self.n_inner_moves += 1
        # The surrogate values ARE the two halves of the correction: for a
        # pi-hat-reversible screen, L(x|x')/Q(x'|x) = pi-hat(x)/pi-hat(x').
        # The q terms cancel and must not be added again here.
        return self._finish(x, ctx, move, prop_move, subtree_root,
                            v_sub_prime, v_sub)

    def __repr__(self):
        return f"<DAProposal ss_prop={self.ss_prop} min_data={self.min_data}>"
