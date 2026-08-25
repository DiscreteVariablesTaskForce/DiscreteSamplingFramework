from discretesampling.base.random import RNG
from discretesampling.domain.incremental_decision_tree.problem import BARRED
from discretesampling.domain.incremental_decision_tree.moves import (
    draw_subtree, evaluate_subtree_move, select_move, subtree_node_data)
from discretesampling.domain.incremental_decision_tree.proposals.base import (
    IncrementalTreeProposalBase)


class IncrementalTreeProposal(IncrementalTreeProposalBase):
    """
    The Metropolis-Hastings proposal for the incremental decision tree domain.
    Draw a subtree, draw a move inside it, apply it,
    and return the new tree and the log proposal correction.
    """

    def sample(self, x, rng=RNG(), target=None):
        return self._timed(self._sample, x, rng)

    def _sample(self, x, rng):
        subtree_root, ctx = draw_subtree(x, rng)
        move, node = select_move(x, ctx, rng)
        if move == "stay":
            return self._stay(x)

        node_data = subtree_node_data(x, ctx, node)
        prop_correction, prop_move = evaluate_subtree_move(
            x, ctx, move, node, node_data, rng)
        if prop_correction <= BARRED:
            # An invalid split, or a reverse path barred at max_tree_size.
            return self._stay(x, 'n_barred')

        self.n_inner_moves += 1
        # forward = 0, reverse = the whole q ratio: the two halves only have to
        # differ by log q(x' -> x) - log q(x -> x'), and putting it all on one
        # side keeps the arithmetic identical to the subsampled proposals'.
        return self._finish(x, ctx, move, prop_move, subtree_root,
                            0.0, prop_correction)
