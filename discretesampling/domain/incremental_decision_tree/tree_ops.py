
import numpy as np
from collections import deque


class IncBDTree:
    """
    This is the base class for the incremental decision tree domain. It is a binary tree
    where each internal node represents a decision based on a feature and a threshold,
    and each leaf node represents a class label. The tree is represented as a list of rows,
    with a record of every data point indice for each leaf, and the total counts of each class for each leaf.

    Attributes
    ----------
    tree      : list of [id, left, right, feat, thr, depth] rows, one per
                internal node
    leaf_idx  : {leaf id: row indices reaching that leaf}
    counts    : {leaf id: class counts, length num_classes}
    problem   : the shared IncrementalTreeProblem, where X, y, and num_classes live
    """

    def __init__(self, problem, tree=None, leaf_idx=None, counts=None,
                 next_id=1, parent_map=None):
        self.problem = problem
        self.tree = [] if tree is None else tree
        self.leaf_idx = {} if leaf_idx is None else leaf_idx
        self.counts = {} if counts is None else counts
        self._next_id = next_id
        self._parent_map = {} if parent_map is None else parent_map
        self._nodes_cache = None
        self._list_nodes_cache = None
        self._terminal_nodes_cache = None

    # -------------------- constructors -------------------- #

    @classmethod
    def stump(cls, problem):
        """
        The simplest tree: a single leaf with all training data routed to it.
        """
        obj = cls(problem)
        obj.leaf_idx[0] = problem.all_rows()
        obj.counts[0] = problem.root_counts()
        return obj

    @classmethod
    def from_rows(cls, problem, rows):
        """
        Construct a tree from a list of rows, as would be returned by encode.
        The rows are assumed to be valid and consistent with the problem's data.
        The leaf membership is re-derived by routing the data, rather than being
        passed in.
        """
        rows = [list(r) for r in rows]
        if not rows:
            return cls.stump(problem)
        obj = cls(problem, tree=rows)
        ids = [int(r[0]) for r in rows]
        children = [int(r[1]) for r in rows] + [int(r[2]) for r in rows]
        obj._next_id = max(ids + children) + 1
        obj._parent_map = {int(r[1]): int(r[0]) for r in rows}
        obj._parent_map.update({int(r[2]): int(r[0]) for r in rows})
        obj._assign_all_leaves()
        return obj

    def _assign_all_leaves(self):
        """
        Build leaf_idx and counts from the tree and the problem's data.
        This is used on decode, where the leaf membership is not shipped.
        """
        node_map = self.nodes
        X, y, K = self.problem.X, self.problem.y, self.problem.num_classes
        self.leaf_idx = {}
        self.counts = {}
        pending = [(0, np.arange(self.problem.n_rows))]
        while pending:
            nid, idx = pending.pop()
            row = node_map.get(nid)
            if row is None:
                self.leaf_idx[nid] = idx
                self.counts[nid] = np.bincount(y[idx], minlength=K)
                continue
            go_left = X[idx, int(row[3])] < row[4]
            pending.append((int(row[1]), idx[go_left]))
            pending.append((int(row[2]), idx[~go_left]))

    # -------------------- cached views -------------------- #

    @property
    def leafs(self):
        return list(self.leaf_idx.keys())

    @property
    def nodes(self):
        """
        {node id: row} for all internal nodes, cached.
        """
        if self._nodes_cache is None:
            self._nodes_cache = {int(r[0]): r for r in self.tree}
        return self._nodes_cache

    @property
    def list_nodes(self):
        if self._list_nodes_cache is None:
            self._list_nodes_cache = [int(r[0]) for r in self.tree]
        return self._list_nodes_cache

    @property
    def terminal_nodes(self):
        """
        Cached set of internal nodes whose children are both leaves.
        these are the only nodes we can prune."""
        if self._terminal_nodes_cache is None:
            leaf_set = set(self.leaf_idx.keys())
            self._terminal_nodes_cache = [
                int(r[0]) for r in self.tree
                if int(r[1]) in leaf_set and int(r[2]) in leaf_set]
        return self._terminal_nodes_cache

    def _invalidate(self):
        self._nodes_cache = None
        self._list_nodes_cache = None
        self._terminal_nodes_cache = None

    # -------------------- descendant helpers -------------------- #

    def _parent_of(self, node_id):
        return self._parent_map.get(node_id)

    def _child_nodes(self, node_id):
        m = self.nodes
        if node_id not in m:
            return []
        return [int(m[node_id][1]), int(m[node_id][2])]

    def _descendant_leaves(self, node_id):
        m = self.nodes
        stack, leaves = [node_id], []
        while stack:
            n = stack.pop()
            if n in m:
                stack.extend([int(m[n][1]), int(m[n][2])])
            else:
                leaves.append(n)
        return leaves

    def _descendant_nodes(self, node_id):
        m = self.nodes
        stack, descendants = [node_id], []
        while stack:
            n = stack.pop()
            if n in m:
                descendants.append(n)
                stack.extend([int(m[n][1]), int(m[n][2])])
        return descendants

    def _descendant_terminal_nodes(self, node_id):
        m = self.nodes
        stack, terminal = [node_id], []
        while stack:
            n = stack.pop()
            if n in m:
                L, R = int(m[n][1]), int(m[n][2])
                if L not in m and R not in m:
                    terminal.append(n)
                else:
                    stack.extend([L, R])
        return terminal

    def _depth(self, node_id):
        """
        Depth of a node in the tree, with the root at depth 0.
        Returns -1 if the node is not in the tree.
        """
        m = self.nodes
        if node_id == 0:
            return 0
        q = deque([(0, 0)])
        visited = {0}
        while q:
            curr, depth = q.popleft()
            if curr not in m:
                continue
            L, R = int(m[curr][1]), int(m[curr][2])
            if L == node_id or R == node_id:
                return depth + 1
            for child in (L, R):
                if child not in visited:
                    q.append((child, depth + 1))
                    visited.add(child)
        return -1

    # -------------------- routing -------------------- #

    def route_one(self, x):
        """
        Determines the leaf id that a single row x reaches,
        by walking the tree from the root.
        Returns the leaf id as an integer.
        """
        m = self.nodes
        nid = 0
        while nid in m:
            _, L, R, feat, thr, _ = m[nid]
            nid = int(L if x[int(feat)] < thr else R)
        return nid

    def route(self, X):
        """
        Determines the leaf id that each row in set X reaches,
        by walking the tree from the root.
        Returns a 1D array of leaf ids, one for each row in X.
        """
        m = self.nodes
        X = np.asarray(X)
        out = np.empty(len(X), dtype=np.int64)
        pending = [(0, np.arange(len(X)))]
        while pending:
            nid, rows = pending.pop()
            if rows.size == 0:
                continue
            row = m.get(nid)
            if row is None:
                out[rows] = nid
                continue
            go_left = X[rows, int(row[3])] < row[4]
            pending.append((int(row[1]), rows[go_left]))
            pending.append((int(row[2]), rows[~go_left]))
        return out

    # -------------------- moves -------------------- #

    def grow(self, leaf_id, feat, thr):
        """
        Grow a leaf into a decision node, splitting on the given feature and threshold.
        The leaf's rows are re-routed to the new children, and the counts are updated.
        """
        if leaf_id not in self.leaf_idx:
            raise ValueError("Chosen node is not a leaf.")
        left_id, right_id = self._next_id, self._next_id + 1
        self._next_id += 2
        depth = self._depth(leaf_id)

        self.tree.append([leaf_id, left_id, right_id, feat, thr, depth])
        self._invalidate()
        self._parent_map[left_id] = leaf_id
        self._parent_map[right_id] = leaf_id

        # Only the rows in this one leaf move.
        idx = np.asarray(self.leaf_idx.pop(leaf_id), dtype=int)
        mask = self.problem.X[idx, feat] < thr
        y, K = self.problem.y, self.problem.num_classes
        for nid, sub_idx in ((left_id, idx[mask]), (right_id, idx[~mask])):
            self.leaf_idx[nid] = sub_idx
            self.counts[nid] = np.bincount(y[sub_idx], minlength=K)
        self.counts.pop(leaf_id, None)
        return self

    def prune(self, parent_id):
        """
        Turn a terminal decision node into a leaf, merging its two children.
        The children must both be leaves, and the parent becomes a leaf with the
        union of the children's rows and counts.
        """
        row = self.nodes.get(parent_id)
        if row is None:
            raise ValueError("Not an internal node.")
        L, R = int(row[1]), int(row[2])
        if L not in self.leaf_idx or R not in self.leaf_idx:
            raise ValueError("Children must both be leaves.")

        self.leaf_idx[parent_id] = np.concatenate(
            [np.asarray(self.leaf_idx.pop(L), dtype=int),
             np.asarray(self.leaf_idx.pop(R), dtype=int)])
        self.counts[parent_id] = self.counts.pop(L) + self.counts.pop(R)

        self.tree = [r for r in self.tree if int(r[0]) != parent_id]
        self._parent_map.pop(L, None)
        self._parent_map.pop(R, None)
        self._invalidate()
        return self

    def change(self, node_id, new_feat, new_thr):
        """
        Change the feature and threshold of a decision node,
        and re-route all the data points that descend from it,
        updating the counts of all affected leaves.
        """
        row = self.nodes.get(node_id)
        if row is None:
            raise ValueError("Not an internal node.")
        row[3], row[4] = new_feat, new_thr

        desc = self._descendant_leaves(node_id)
        idx_all = (np.concatenate([np.asarray(self.leaf_idx.get(leaf, []), dtype=int)
                                   for leaf in desc])
                   if desc else np.empty(0, dtype=int))
        for leaf in desc:
            self.leaf_idx.pop(leaf, None)
            self.counts.pop(leaf, None)

        m = self.nodes
        X, y, K = self.problem.X, self.problem.y, self.problem.num_classes
        queue = deque([(node_id, idx_all)])
        while queue:
            curr, curr_idx = queue.popleft()
            if curr in m:
                _, L, R, feat, thr, _ = m[curr]
                mask = X[curr_idx, int(feat)] < thr
                queue.append((int(L), curr_idx[mask]))
                queue.append((int(R), curr_idx[~mask]))
            else:
                self.leaf_idx[curr] = curr_idx
                self.counts[curr] = np.bincount(y[curr_idx], minlength=K)
        return self

    # -------------------- copying -------------------- #

    def deep_copy(self):
        """
        Return a deep copy of this tree, with all mutable attributes copied.
        The new tree shares the same problem object, but has its own tree structure,
        leaf membership, and counts.
        """
        new = self.__class__.__new__(self.__class__)
        new.problem = self.problem
        new.tree = [row.copy() for row in self.tree]
        new.leaf_idx = dict(self.leaf_idx)
        new.counts = dict(self.counts)
        new._next_id = self._next_id
        new._parent_map = self._parent_map.copy()
        new._nodes_cache = None
        new._list_nodes_cache = None
        new._terminal_nodes_cache = None
        return new
