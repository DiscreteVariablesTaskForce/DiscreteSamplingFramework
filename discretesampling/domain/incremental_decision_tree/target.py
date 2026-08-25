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

    def _terminal_node_likelihood(self, thr, feat, idx, scale=1.0):
        left = self.problem.X[idx, feat] < thr
        sub_y = self.problem.y[idx]
        K = self.problem.num_classes
        return (self.log_dm(np.bincount(sub_y[left], minlength=K) * scale)
                + self.log_dm(np.bincount(sub_y[~left], minlength=K) * scale))

    def _leaf_node_likelihood(self, idx, scale=1.0):
        return self.log_dm(np.bincount(self.problem.y[idx],
                                       minlength=self.problem.num_classes) * scale)

    def _general_node_likelihood(self, thr, feat, state, node, idx,
                                 scale=1.0, min_leaf=0):
        """
        Calculates the summed Dirichlet-Multinomial likelihood of all
        descendant leaves of given node.
        """
        total = 0.0
        K = self.problem.num_classes
        X, y = self.problem.X, self.problem.y
        queue = deque([(node, idx, True)])

        while queue:
            curr, curr_idx, is_changed = queue.popleft()
            children = state._child_nodes(curr)
            if (not children) or curr in state.leaf_idx or len(children) != 2:
                if min_leaf and len(curr_idx) < min_leaf:
                    return None
                total += self.log_dm(np.bincount(y[curr_idx], minlength=K) * scale)
                continue

            node_feat, node_thr = state.nodes[curr][3:5]
            curr_feat = int(feat) if is_changed else int(node_feat)
            curr_thr = float(thr) if is_changed else float(node_thr)
            left = X[curr_idx, curr_feat] < curr_thr
            queue.append((children[0], curr_idx[left], False))
            queue.append((children[1], curr_idx[~left], False))

        return total

    def eval_move(self, state, move, node, prop_move, idx, scale=1.0, min_leaf=0):
        """
        The surrogate move ratio on one block of rows: (v, v_prime).
        The surrogate is the log-target density of the subtree rooted at `node`,
        evaluated on the current tree and on the proposed tree.
        The subtree is evaluated on the given `idx` of rows, scaled by `scale`
        to make the scaled counts sum to the node's true row count when using a subset of rows.
        """
        problem = self.problem
        delta = {"grow": 1, "prune": -1}.get(move, 0)
        n_nodes, n_leaves = len(state.tree), len(state.leaf_idx)

        prior = self.log_prior(n_nodes, n_leaves)
        prior_prime = self.log_prior(n_nodes + delta, n_leaves + delta)

        row = state.nodes.get(node)
        old_feat = int(row[3]) if row is not None else 0
        old_thr = float(row[4]) if row is not None else 0.0
        lp_old = problem.lp_feats + problem.lp_vals[old_feat]

        if move == "prune":
            v = self._terminal_node_likelihood(old_thr, old_feat, idx, scale) + lp_old
            v_prime = self._leaf_node_likelihood(idx, scale)

        elif move == "change":
            v = self._general_node_likelihood(old_thr, old_feat, state, node, idx,
                                              scale) + lp_old
            lhood_new = self._general_node_likelihood(
                float(prop_move['thr']), int(prop_move['feat']), state, node, idx,
                scale, min_leaf=min_leaf)
            if lhood_new is None:
                # Returned bare, not offset by prior_prime: callers test it
                # against the large_neg sentinel.
                return v + prior, large_neg
            v_prime = (lhood_new + problem.lp_feats
                       + problem.lp_vals[int(prop_move['feat'])])

        else:  # grow
            v = self._leaf_node_likelihood(idx, scale)
            v_prime = (self._terminal_node_likelihood(
                float(prop_move['thr']), int(prop_move['feat']), idx, scale)
                + problem.lp_feats + problem.lp_vals[int(prop_move['feat'])])

        return v + prior, v_prime + prior_prime

    def eval_subset(self, state, move, node, prop_move, subset, size):
        """
        The surrogate move ratio on a subset of rows: (v, v_prime, success).
        The surrogate is the log-target density of the subtree rooted at `node`,
        evaluated on the current tree and on the proposed tree.
        The subtree is evaluated on the given `subset` of rows, scaled by size/len(subset)
        to make the scaled counts sum to the node's true row count.
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
        v, v_prime = self.eval_move(state, move, node, prop_move, subset, scale)
        return v, v_prime, True
