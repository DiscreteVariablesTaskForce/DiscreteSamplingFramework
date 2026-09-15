from collections import deque
from math import log, lgamma

import numpy as np
from scipy.special import gammaln

from discretesampling.base.types import DiscreteVariableTarget
from discretesampling.domain.incremental_decision_tree.problem import large_neg
from discretesampling.domain.incremental_decision_tree.moves import subtree_admissible

_log_catalan_cache = {}


def log_catalan(n_leaves):
    """
    Prior on tree shape:
    The log of the m-th Catalan number, m = n_leaves - 1: the number of binary
    tree shapes with this many leaves, which the structural prior divides out.
    """
    if n_leaves <= 1:
        return 0.0
    value = _log_catalan_cache.get(n_leaves)
    if value is None:
        m = n_leaves - 1
        value = lgamma(2 * m + 1) - 2 * lgamma(m + 1) - log(m + 1)
        _log_catalan_cache[n_leaves] = value
    return value


def log_poisson(n_nodes, lam):
    return n_nodes * log(lam) - lam - gammaln(n_nodes + 1)


class IncrementalTreeTarget(DiscreteVariableTarget):

    def __init__(self, problem):
        self.problem = problem
        # Instrumentation: what the comparison between the three methods is
        # actually counting. Full evaluations are the expensive ones.
        self.n_full_evals = 0
        self.n_memo_hits = 0
        self.n_subset_evals = 0

    def reset_counters(self):
        self.n_full_evals = self.n_memo_hits = self.n_subset_evals = 0

    def counters(self):
        return {'full_evals': self.n_full_evals,
                'memo_hits': self.n_memo_hits,
                'subset_evals': self.n_subset_evals}

    # ------------------- the exact target ------------------- #

    def eval(self, x):
        """
        Evaluate the log-target density of `x`, including:
        - the structural prior on the number of nodes and leaves,
        - the uniform priors on the features and thresholds of every internal node,
        - the Dirichlet-Multinomial likelihood of every leaf.
        Returns a large negative number when `x` is not valid,
        e.g. when any leaf holds fewer rows than min_samples_leaf.
        """
        cached = getattr(x, '_log_target', None)
        if cached is not None:
            self.n_memo_hits += 1
            return cached
        self.n_full_evals += 1

        leaves = x.leafs
        lhood = self.sum_leaf_log_dm(x, leaves,
                                     min_leaf=self.problem.min_samples_leaf)
        if lhood is None:
            value = large_neg
        else:
            value = (self.log_prior(len(x.tree), len(leaves))
                     + self.sum_split_priors(x, x.list_nodes)
                     + lhood)
        x._log_target = value
        return value

    def evaluatePrior(self, x):
        """
        The log-prior density of `x`, including:
        - the structural prior on the number of nodes and leaves,
        - the uniform priors on the features and thresholds of every internal node.
        """
        return (self.log_prior(len(x.tree), len(x.leaf_idx))
                + self.sum_split_priors(x, x.list_nodes))

    def log_prior(self, n_nodes, n_leaves):
        """
        The log of the prior on tree shape and size:
        - a Poisson prior on the number of nodes.
        - a Catalan prior on the number of leaves.
        """
        return log_poisson(n_nodes, self.problem.lam) - log_catalan(n_leaves)

    def sum_split_priors(self, state, nodes):
        """
        The prior of a tree's splits, summed over the given nodes.
        Each internal node contributes a uniform prior on its feature and threshold.
        """
        if not nodes:
            return 0.0
        problem = self.problem
        return (len(nodes) * problem.lp_feats
                + sum(problem.lp_vals[int(state.nodes[n][3])] for n in nodes))

    # ------------------- Dirichlet-Multinomial ------------------- #

    def log_dm(self, counts):
        p = self.problem
        return (p._lgamma_a0 - gammaln(p._a0 + counts.sum())
                + gammaln(counts + p.alpha).sum() - p._K_lgamma_alpha)

    def sum_leaf_log_dm(self, state, leaves, min_leaf=0):
        """
        Sum of the Log Dirichlet-Multinomial likelihoods of the given leaves.
        Returns None if any leaf has fewer than `min_leaf` rows.
        """
        if not leaves:
            return 0.0
        p = self.problem
        C = np.stack([state.counts[leaf] for leaf in leaves])
        n = C.sum(axis=1)
        if min_leaf > 0 and n.min() < min_leaf:
            return None
        L = len(leaves)
        return (L * p._lgamma_a0 - gammaln(p._a0 + n).sum()
                + gammaln(C + p.alpha).sum() - L * p._K_lgamma_alpha)

    def subtree_admissible(self, state, subtree_root):
        """
        Returns True if the subtree rooted at `subtree_root` is admissible,
        i.e. every leaf in the subtree holds at least min_samples_leaf rows.
        """
        return subtree_admissible(state, subtree_root)

    # ------------------- local move likelihoods ------------------- #
    #
    # A move's surrogate is scored on a subset of the rows under `node`, passed
    # grouped by the leaf of the current tree each row sits in: leaf_pos[i] is
    # the position in `leaves` (every leaf under `node`) of subset[i]'s leaf.
    # That grouping already is the current tree's side of every move, so it is
    # never routed again; only a change routes, and only the rows its new split
    # sends across. The scores are the ones routing every subset row from
    # `node` gives: the same counts, summed leaf by leaf in the same order.

    def _leaf_counts(self, y_sub, leaf_pos, n_leaves):
        """(n_leaves, K) class counts of subset rows, by the leaf they sit in."""
        K = self.problem.num_classes
        return np.bincount(leaf_pos * K + y_sub,
                           minlength=n_leaves * K).reshape(n_leaves, K)

    def _sum_leaf_log_dm(self, counts, order, scale):
        """log_dm of each row of counts * scale, summed over the rows in `order`."""
        p = self.problem
        scaled = counts * scale
        per_leaf = (p._lgamma_a0 - gammaln(p._a0 + scaled.sum(axis=1))
                    + gammaln(scaled + p.alpha).sum(axis=1) - p._K_lgamma_alpha)
        total = 0.0
        for k in order:
            total += per_leaf[k]
        return total

    @staticmethod
    def _breadth_first_positions(state, node, leaves):
        """Positions in `leaves` of the leaves under `node`, breadth first from it."""
        m = state.nodes
        pos = {leaf: k for k, leaf in enumerate(leaves)}
        order, queue = [], deque([node])
        while queue:
            curr = queue.popleft()
            row = m.get(curr)
            if row is None:
                order.append(pos[curr])
            else:
                queue.append(int(row[1]))
                queue.append(int(row[2]))
        return order

    def _changed_counts(self, state, node, feat, thr, subset, y_sub, leaf_pos,
                        leaves, counts):
        """
        The subset's leaf counts once `node` splits on (feat, thr) instead.
        Nothing below `node` changes, so a row that lands on the same side as
        before keeps its leaf; only the rows sent across are routed, down the
        other child's subtree.
        """
        problem, m = self.problem, state.nodes
        X, y, K = problem.X, problem.y, problem.num_classes
        left_child, right_child = int(m[node][1]), int(m[node][2])
        left_leaves = set(state._descendant_leaves(left_child))
        was_left = np.array([leaf in left_leaves for leaf in leaves])[leaf_pos]
        goes_left = X[subset, feat] < thr
        crossed = was_left != goes_left
        new = counts - self._leaf_counts(y_sub[crossed], leaf_pos[crossed], len(leaves))

        pos = {leaf: k for k, leaf in enumerate(leaves)}
        for child, moving in ((left_child, crossed & goes_left),
                              (right_child, crossed & ~goes_left)):
            queue = deque([(child, subset[moving])])
            while queue:
                curr, curr_idx = queue.popleft()
                if curr_idx.size == 0:
                    continue
                row = m.get(curr)
                if row is None:
                    new[pos[curr]] += np.bincount(y[curr_idx], minlength=K)
                    continue
                go_left = X[curr_idx, int(row[3])] < float(row[4])
                queue.append((int(row[1]), curr_idx[go_left]))
                queue.append((int(row[2]), curr_idx[~go_left]))
        return new

    def eval_move(self, state, move, node, prop_move, subset, leaf_pos, leaves,
                  scale=1.0):
        """
        The surrogate move ratio on one block of rows: (v, v_prime).
        The surrogate is the log-target density of the subtree rooted at `node`,
        evaluated on the current tree and on the proposed tree.
        The subtree is evaluated on the rows `subset`, grouped by leaf as above,
        scaled by `scale` to make the scaled counts sum to the node's true row
        count when using a subset of rows.
        """
        problem = self.problem
        delta = {"grow": 1, "prune": -1}.get(move, 0)
        n_nodes, n_leaves = len(state.tree), len(state.leaf_idx)

        prior = self.log_prior(n_nodes, n_leaves)
        prior_prime = self.log_prior(n_nodes + delta, n_leaves + delta)

        row = state.nodes.get(node)
        old_feat = int(row[3]) if row is not None else 0
        lp_old = problem.lp_feats + problem.lp_vals[old_feat]
        y_sub = problem.y[subset]

        if move == "prune":
            # A terminal node's children are both leaves, so the leaf a row
            # sits in is the side of the split it goes to.
            counts = self._leaf_counts(y_sub, leaf_pos, len(leaves))
            left = counts[leaves.index(int(row[1]))]
            right = counts[leaves.index(int(row[2]))]
            v = self.log_dm(left * scale) + self.log_dm(right * scale) + lp_old
            v_prime = self.log_dm((left + right) * scale)

        elif move == "change":
            counts = self._leaf_counts(y_sub, leaf_pos, len(leaves))
            order = self._breadth_first_positions(state, node, leaves)
            feat, thr = int(prop_move['feat']), float(prop_move['thr'])
            v = self._sum_leaf_log_dm(counts, order, scale) + lp_old
            new = self._changed_counts(state, node, feat, thr, subset, y_sub,
                                       leaf_pos, leaves, counts)
            v_prime = (self._sum_leaf_log_dm(new, order, scale)
                       + problem.lp_feats + problem.lp_vals[feat])

        else:  # grow: `node` is a leaf
            K = problem.num_classes
            feat = int(prop_move['feat'])
            left = problem.X[subset, feat] < float(prop_move['thr'])
            v = self.log_dm(np.bincount(y_sub, minlength=K) * scale)
            v_prime = (self.log_dm(np.bincount(y_sub[left], minlength=K) * scale)
                       + self.log_dm(np.bincount(y_sub[~left], minlength=K) * scale)
                       + problem.lp_feats + problem.lp_vals[feat])

        return v + prior, v_prime + prior_prime

    def eval_subset(self, state, move, node, prop_move, subset, leaf_pos, leaves,
                    size):
        """
        The surrogate move ratio on a subset of rows: (v, v_prime, success).
        The surrogate is the log-target density of the subtree rooted at `node`,
        evaluated on the current tree and on the proposed tree.
        The subtree is evaluated on the rows `subset`, where leaf_pos[i] is the
        position in `leaves` -- every leaf under `node` -- of the leaf subset[i]
        sits in, scaled by size/len(subset) to make the scaled counts sum to
        the node's true row count.
        Returns success=False when the subset is empty, which means the move
        cannot be evaluated and should be rejected rather than guessed at.
        """
        if move == "stay":
            return 0.0, 0.0, True
        if subset is None or len(subset) == 0:
            # No row of this block reaches the node: the surrogate has no
            # information, so the move is rejected rather than guessed at.
            return 0.0, large_neg, False
        self.n_subset_evals += 1
        scale = size / len(subset)
        v, v_prime = self.eval_move(state, move, node, prop_move, subset, leaf_pos,
                                    leaves, scale)
        return v, v_prime, True
