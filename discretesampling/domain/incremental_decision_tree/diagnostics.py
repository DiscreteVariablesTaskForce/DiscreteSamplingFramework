"""
Per-move instrumentation for the incremental decision tree proposals.

The three proposals do different amounts of work per iteration -- MH considers
one move, DA considers one and screens it against a surrogate on one subsample,
HINTS considers one per block of a partition and screens each -- so comparing
them has to happen at the level of the individual move, not the iteration. A
MoveLog holds one row per move considered, tagged with the sample() call it came
from, plus one row per call for the fate of that call as a whole.

Recording is off by default (`proposal.record_moves`); when off a move costs one
attribute load and a branch. When on it costs seven appends of scalars to plain
Python lists -- no dicts, no per-move objects. The lists become numpy arrays
only in arrays(), once, at the end of a run.

Reading the two levels together
-------------------------------
The per-move outcome is the fate of the move *inside* the proposal. It is not
the accept/reject of the sampler:

  * MH and DA emit one move row per call. PROPOSED means the move survived the
    proposal and was handed to the outer MH step; whether that step took it is
    the `accepted` flag the MCMC callback reports for the same call index.
  * HINTS emits one row per block. APPLIED means the move was accepted by its
    block's surrogate and written into the sweep's working tree. The whole sweep
    is then a single proposal, so those APPLIED moves survive only if the call
    row says PROPOSED *and* the outer step accepted that call.

So a HINTS move is accepted iff  move.outcome == APPLIED  and
call_outcome[call] == PROPOSED  and  accepted[call].
"""
import numpy as np

# Move types, in the order moves.MOVE_PROBS lists them, with "stay" prepended
# for the case where no valid node could be drawn.
MOVES = ("stay", "grow", "prune", "change")
MOVE_CODE = {name: i for i, name in enumerate(MOVES)}

# What became of a move inside the proposal.
STAY = 0            # no valid move was available
BARRED = 1          # the proposal correction was BARRED (invalid split, or a
#                     reverse path barred at max_tree_size)
SCREENED_OUT = 2    # rejected by the surrogate on a subsample (DA, HINTS)
INADMISSIBLE = 3    # rejected by a min_samples_leaf admissibility check
PROPOSED = 4        # survived the proposal, handed to the outer MH step
APPLIED = 5         # written into a HINTS sweep's working tree
OUTCOMES = ("stay", "barred", "screened_out", "inadmissible", "proposed",
            "applied")

# The counter name each _stay() reason maps onto.
STAY_OUTCOME = {None: STAY, 'n_barred': BARRED, 'n_screened_out': SCREENED_OUT,
                'n_inadmissible': INADMISSIBLE}


class MoveLog:
    """
    An append-only record of every move a proposal considers.

    Columns, one row per move:
        call     index of the sample() call the move belongs to
        move     MOVE_CODE of the move type
        node     the node it acted on, or -1 for a "stay"
        rows     rows under that node in the full training set
        subset   rows of the surrogate subsample the move was screened on
                 (0 when it was not screened, as in MH)
        dsurr    the surrogate move ratio v' - v that the screen used
                 (nan when there was no screen)
        outcome  one of the outcome codes above

    And one row per sample() call:
        call_id, call_outcome
    """

    __slots__ = ('call', 'move', 'node', 'rows', 'subset', 'dsurr', 'outcome',
                 'call_id', 'call_outcome')

    def __init__(self):
        self.clear()

    def clear(self):
        self.call = []
        self.move = []
        self.node = []
        self.rows = []
        self.subset = []
        self.dsurr = []
        self.outcome = []
        self.call_id = []
        self.call_outcome = []

    def __len__(self):
        return len(self.move)

    def record(self, call, move, node, rows, subset, dsurr, outcome):
        self.call.append(call)
        self.move.append(move)
        self.node.append(node)
        self.rows.append(rows)
        self.subset.append(subset)
        self.dsurr.append(dsurr)
        self.outcome.append(outcome)

    def end_call(self, call, outcome):
        self.call_id.append(call)
        self.call_outcome.append(outcome)

    def arrays(self):
        """The log as numpy arrays, ready to go into an .npz."""
        return {
            'call': np.asarray(self.call, dtype=np.int64),
            'move': np.asarray(self.move, dtype=np.int8),
            'node': np.asarray(self.node, dtype=np.int64),
            'rows': np.asarray(self.rows, dtype=np.int64),
            'subset': np.asarray(self.subset, dtype=np.int64),
            'dsurr': np.asarray(self.dsurr, dtype=np.float64),
            'outcome': np.asarray(self.outcome, dtype=np.int8),
            'call_id': np.asarray(self.call_id, dtype=np.int64),
            'call_outcome': np.asarray(self.call_outcome, dtype=np.int8),
        }

    def flush(self):
        """arrays(), then start again empty -- for reducing a long run in
        pieces instead of holding every move of it in memory."""
        out = self.arrays()
        self.clear()
        return out

    def counts(self):
        """
        Move type against outcome, as a (len(MOVES), len(OUTCOMES)) count array.
        This is the headline table for comparing proposals: it says how a
        proposal's moves are being spent -- and, for DA and HINTS, how much of
        the work the surrogate screen is throwing away before the full target is
        ever touched.
        """
        table = np.zeros((len(MOVES), len(OUTCOMES)), dtype=np.int64)
        if not self.move:
            return table
        flat = np.bincount(
            np.asarray(self.move, dtype=np.int64) * len(OUTCOMES)
            + np.asarray(self.outcome, dtype=np.int64),
            minlength=table.size)
        return flat.reshape(table.shape)

    def format_counts(self):
        """counts() as a printable table."""
        table = self.counts()
        head = "%-9s" % "move" + "".join("%14s" % o for o in OUTCOMES) + "%14s" % "total"
        lines = [head, "-" * len(head)]
        for i, name in enumerate(MOVES):
            lines.append("%-9s" % name
                         + "".join("%14d" % c for c in table[i])
                         + "%14d" % table[i].sum())
        lines.append("%-9s" % "total"
                     + "".join("%14d" % c for c in table.sum(axis=0))
                     + "%14d" % table.sum())
        return "\n".join(lines)
