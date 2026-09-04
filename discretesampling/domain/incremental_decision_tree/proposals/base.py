import time

from discretesampling.base.random import RNG
from discretesampling.base.types import DiscreteVariableProposal
from discretesampling.domain.incremental_decision_tree.problem import BARRED
from discretesampling.domain.incremental_decision_tree.moves import (
    apply_subtree_proposal, proposal_root_correction, root_correction,
    subtree_admissible)
from discretesampling.domain.incremental_decision_tree.diagnostics import (
    MOVE_CODE, MoveLog, PROPOSED, STAY_OUTCOME)

NAN = float('nan')


class IncrementalTreeProposalBase(DiscreteVariableProposal):
    """
    Base class for proposals in the incremental decision tree domain.
    """

    def __init__(self):
        # Off by default: writing it costs a dict per accepted move, which the
        # tests want and a production run does not.
        self.record_diagnostics = False
        # Off by default too, and separately: this one is per move *considered*,
        # not per move accepted, and it is what a sampler comparison reads.
        self.record_moves = False
        self.move_log = MoveLog()
        # The move this sample() call is working on, held until whichever exit
        # the call takes says what became of it. See diagnostics.MoveLog.
        self._pending = None
        self.n_calls = 0
        self.n_stays = 0
        self.n_barred = 0
        self.n_screened_out = 0
        self.n_inadmissible = 0
        self.n_moved = 0
        self.n_inner_moves = 0
        self.sample_time = 0.0

    # ------------------- instrumentation ------------------- #

    def reset_counters(self):
        for name in ('n_calls', 'n_stays', 'n_barred', 'n_screened_out',
                     'n_inadmissible', 'n_moved', 'n_inner_moves'):
            setattr(self, name, 0)
        self.sample_time = 0.0
        self.move_log.clear()
        self._pending = None

    # The call index a move belongs to is n_calls, which _timed increments only
    # once the call has returned -- so during a call it is this call's index.

    def _note(self, move, node, rows, subset=0, dsurr=NAN):
        """
        Remember the move now under consideration. Called again for the same
        move once the surrogate has been evaluated, to fill in the subsample it
        was screened on; the last note before the exit is the one recorded.
        """
        if self.record_moves:
            self._pending = (MOVE_CODE[move], -1 if node is None else node,
                             rows, subset, dsurr)

    def _emit(self, outcome):
        """Record the pending move under `outcome`, if there is one."""
        pending = self._pending
        if pending is not None:
            self.move_log.record(self.n_calls, *pending, outcome)
            self._pending = None

    def _log(self, move, node, rows, outcome, subset=0, dsurr=NAN):
        """Note and emit in one step, for a move whose fate is already known --
        which is every move inside a HINTS sweep, since the sweep decides each
        one before going on to the next."""
        if self.record_moves:
            self.move_log.record(self.n_calls, MOVE_CODE[move],
                                 -1 if node is None else node, rows, subset,
                                 dsurr, outcome)

    def counters(self):
        """
        Collect the proposal's counters into a dict for reporting.
        """
        return {'calls': self.n_calls,
                'stays': self.n_stays,
                'barred': self.n_barred,
                'screened_out': self.n_screened_out,
                'inadmissible': self.n_inadmissible,
                'moved': self.n_moved,
                'inner_moves': self.n_inner_moves,
                'sample_time': self.sample_time}

    # ------------------- the DiscreteVariableProposal interface ------------------- #

    @classmethod
    def norm(cls, tree):
        return len(tree.tree)

    @classmethod
    def heuristic(cls, x, y):
        """
        A heuristic for the proposal's norm, used to scale the proposal's
        log-density when the particle is a subtree of the full tree.
        The heuristic is a lower bound on the number of nodes in the full tree, so
        the proposal's log-density is scaled down to make the proposal more
        conservative when the particle is a subtree.
        """
        return True

    def sample(self, x, rng=RNG(), target=None):
        raise NotImplementedError

    def eval(self, x, x_prime, target=None):
        """
        Evaluate the log-density of the proposal for a move from x to x_prime.
        The proposal's log-density is recorded when the move is drawn, so this
        """
        terms = getattr(x_prime, '_smc_terms', None)
        if terms is not None and terms[0] is x:
            return terms[1]
        terms = getattr(x, '_smc_terms', None)
        if terms is not None and terms[0] is x_prime:
            return terms[2]
        raise ValueError(
            f"{type(self).__name__}.eval was asked about a transition this "
            "proposal did not produce. The subsampled correction is "
            "path-dependent and is recorded by sample(); it cannot be "
            "recomputed for an arbitrary pair of particles. In particular, "
            "use_optimal_L=True is not supported with this proposal.")

    def lkernel(self):
        """
        Returns the reversal of the proposal.
        """
        return IncrementalTreeLKernel(self)

    # ------------------- ending a sample() call ------------------- #

    def _stay(self, x, reason=None):
        """
        Return the original particle, recording that it was not moved.
        The reason is recorded in a counter if given.
        """
        if reason is not None:
            setattr(self, reason, getattr(self, reason) + 1)
        self.n_stays += 1
        if self.record_moves:
            # Every path that gives up on a call ends here, so this is where the
            # pending move's fate is settled and the call is closed out.
            outcome = STAY_OUTCOME[reason]
            self._emit(outcome)
            self.move_log.end_call(self.n_calls, outcome)
        x._smc_terms = (x, 0.0, 0.0)
        return x

    def _finish(self, x, ctx, move, prop_move, subtree_root, forward, reverse):
        """
        Finish a sample() call by applying the move to a copy of x and recording
        the correction. The move is applied to a copy so that the original x is
        not modified if the move is rejected by the target.
        """
        if proposal_root_correction(x, move, prop_move['node'], subtree_root) <= BARRED:
            return self._stay(x, 'n_barred')
        new = x.deep_copy()
        apply_subtree_proposal(new, ctx, move, prop_move)
        return self._finish_sweep(x, new, subtree_root, forward, reverse)

    def _finish_sweep(self, x, state_new, subtree_root, forward, reverse):
        """
        Finish a sample() call by evaluating the subtree's log-target density on
        the original and proposed trees, and recording the correction. The move is
        applied to a copy so that the original x is not modified if the move is rejected by the target.
        """
        if state_new is x:
            return self._stay(x)
        if not subtree_admissible(state_new, subtree_root):
            return self._stay(x, 'n_inadmissible')
        root_term = root_correction(x, state_new, subtree_root)
        if root_term <= BARRED:
            return self._stay(x, 'n_barred')

        self.n_moved += 1
        if self.record_moves:
            # The one exit that does not go through _stay: the proposal has
            # produced a new state and the outer accept/reject step decides it.
            self._emit(PROPOSED)
            self.move_log.end_call(self.n_calls, PROPOSED)
        state_new._smc_terms = (x, float(forward), float(reverse + root_term))
        if self.record_diagnostics:
            # The terms split out, for tests that check the correction against
            # an independently recomputed one.
            state_new._smc_diag = {'subtree_root': subtree_root,
                                   'root_term': float(root_term),
                                   'forward': float(forward),
                                   'reverse': float(reverse)}
        return state_new

    # ------------------- timing ------------------- #

    def _timed(self, fn, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.sample_time += time.perf_counter() - start
            self.n_calls += 1


class IncrementalTreeLKernel(DiscreteVariableProposal):
    """
    The reversal of an IncrementalTreeProposalBase.
    """

    def __init__(self, proposal):
        self.proposal = proposal

    @classmethod
    def norm(cls, tree):
        return len(tree.tree)

    @classmethod
    def heuristic(cls, x, y):
        return True

    def sample(self, x, rng=RNG(), target=None):
        raise NotImplementedError(
            "An L-kernel is only ever evaluated, never sampled from.")

    def eval(self, x_prime, x, target=None):
        return self.proposal.eval(x_prime, x)

    def __repr__(self):
        return f"<IncrementalTreeLKernel of {type(self.proposal).__name__}>"
