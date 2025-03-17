# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:32:34 2025

@author: efthi
"""

import numpy as np
from collections import deque
from collections import defaultdict
import pandas as pd
import copy
from sklearn.base import BaseEstimator, ClassifierMixin

def perturb_data(X, epsilon=0.3, feature_idx=None, seed=None):
    """
    Apply random perturbations to the test data with a fixed random seed for reproducibility.
    :param X: Input features.  
    :param epsilon: Perturbation magnitude.
    :param feature_idx: Optional list of indices to perturb specific features.
    :param seed: Optional random seed for reproducibility.
    :return: Perturbed data.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility

    X_adv = np.copy(X)
    n_samples, n_features = X.shape
    
    # Perturb either specific feature indices or all features
    if feature_idx is None:
        feature_idx = range(n_features)
    
    for i in range(n_samples):
        for j in feature_idx:
            perturbation = np.random.uniform(-epsilon, epsilon)  # Small random perturbation
            X_adv[i, j] += perturbation
            # Ensure that the perturbed value remains within valid feature bounds (e.g., non-negative)
            X_adv[i, j] = np.clip(X_adv[i, j], 0, None)  # Clip to ensure valid range (optional)
    
    return X_adv

def restructure_tree(original_tree):

    # Extract all node indices (parent, left, right) that are used in the tree
    used_indices = set()
    for node in original_tree:
        used_indices.update(node[:3])  # Parent, left child, right child
    
    # Sort indices to determine the new sequential numbering
    used_indices = sorted(used_indices)
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    
    # Restructure the tree
    restructured_tree = []
    for node in original_tree:
        parent, left, right, split_feature, split_value, depth = node
        new_node = [
            index_mapping[parent],  # Update parent index
            index_mapping[left],    # Update left child index
            index_mapping[right],   # Update right child index
            split_feature,          # Split feature remains the same
            split_value,            # Split value remains the same
            depth                   # Depth remains the same
        ]
        restructured_tree.append(new_node)
        
    return restructured_tree

def preprocess_tree(tree):
    """Extract parent nodes, child nodes, and the maximum node ID."""
    parent_nodes = set()
    child_nodes = set()
    max_node_id = -1

    for entry in tree:
        parent_nodes.add(entry[0])
        child_nodes.add(entry[1])
        child_nodes.add(entry[2])
        max_node_id = max(max_node_id, entry[0], entry[1], entry[2])

    return parent_nodes, child_nodes, max_node_id

def get_leaf_nodes(parent_nodes, child_nodes):
    """Compute leaf nodes from parent and child nodes."""
    # Leaf nodes are child nodes that are not parent nodes
    leaf_nodes = child_nodes - parent_nodes
    
    return sorted(leaf_nodes)# Convert to a sorted list for readability

def get_threshold_and_feature_lists(tree, max_node_id):
    """Create threshold and feature lists for all nodes."""
    # Initialize the lists with -2
    threshold_list = [-2] * (max_node_id + 1)
    feature_list = [-2] * (max_node_id + 1)

    # Populate the lists with thresholds and features from the tree
    for entry in tree:
        node_id = entry[0]
        threshold = entry[4]
        feature = entry[3]
        threshold_list[node_id] = threshold
        feature_list[node_id] = feature

    return threshold_list, feature_list

def find_left_right_chilren(tree, max_node_id):
    """Populate the left and right child lists."""
    # Initialize both lists with -1, which is the default for leaf nodes
    left_child = [-1] * (max_node_id + 1)
    right_child = [-1] * (max_node_id + 1)

    # Loop through each node in the tree to populate the lists
    for entry in tree:
        parent = entry[0]
        left = entry[1]
        right = entry[2]
        
        left_child[parent] = left  # Assign the left child
        right_child[parent] = right  # Assign the right child
    
    return left_child, right_child

def compute_values(X_train, y_train, feature_list, threshold_list, left_child, right_child):
    """
    Compute label distributions for each node in the tree (no pruning here).
    Keeps labels as-is without remapping to zero-indexed.
    """
    # Convert to numpy if using pandas
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, (pd.Series, pd.DataFrame)):
        y_train = y_train.values.ravel()

    # Unique labels
    unique_labels = np.unique(y_train)
    num_labels = len(unique_labels)
    n_nodes = len(feature_list)
    values = [np.zeros((1, num_labels)) for _ in range(n_nodes)]  # Store [ [p_class_label1, p_class_label2, ...] ]

    def traverse(node_id, subset):
        if node_id == -1 or node_id >= n_nodes:
            return

        # Count how many of each label in this subset
        labels, counts = np.unique(subset[:, -1], return_counts=True)
        distribution = np.zeros(num_labels)
        for label, c in zip(labels, counts):
            label_index = np.where(unique_labels == label)[0][0]  # Map to index
            distribution[label_index] = c

        total = distribution.sum()
        if total > 0:
            values[node_id][0] = distribution / total
        # else remains zeros

        # Check if leaf or invalid feature => stop
        feat = feature_list[node_id]
        if feat == -2 or (left_child[node_id] == -1 and right_child[node_id] == -1):
            return
        if feat < 0 or feat >= X_train.shape[1]:
            return  # Treat as leaf as well

        # Split
        thr = threshold_list[node_id]
        left_mask = subset[:, feat] <= thr
        right_mask = ~left_mask

        left_data = subset[left_mask]
        right_data = subset[right_mask]

        traverse(left_child[node_id], left_data)
        traverse(right_child[node_id], right_data)

    # Start at root
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    traverse(0, data)
    return values, unique_labels  # Return values and original labels




def traverse_subtree(node_id, left_child, right_child, subtree_nodes, leaf_nodes):
    """Recursive function to traverse the subtree from node_id."""
    # If node_id is invalid, just return
    if node_id == -1:
        return
    
    # Add the current node to the subtree list
    subtree_nodes.append(node_id)
    
    # Check if current node is a leaf
    if left_child[node_id] == -1 and right_child[node_id] == -1:
        leaf_nodes.append(node_id)
    
    # Recurse on left and right children
    traverse_subtree(left_child[node_id], left_child, right_child, subtree_nodes, leaf_nodes)
    traverse_subtree(right_child[node_id], left_child, right_child, subtree_nodes, leaf_nodes)

def get_subtree_info(start_node, left_child, right_child):
    """Get all nodes and leaf nodes in the subtree rooted at start_node."""
    subtree_nodes = []
    leaf_nodes = []
    traverse_subtree(start_node, left_child, right_child, subtree_nodes, leaf_nodes)
    return subtree_nodes, leaf_nodes

def prune_tree_with_purity(feature_list, threshold_list, left_child, right_child, values, Leaf_nodes_pr):
  
    # Find arrays with at least one element equal to 1
    indices_with_one = [idx for idx, arr in enumerate(values) if np.any(arr == 1)]
    zero_sum_nodes = [idx for idx, val in enumerate(values) if np.isclose(np.sum(val), 0)]
    # print("indices_with_one: ", indices_with_one)
    # print("zero_sum_nodes: ", zero_sum_nodes)
    
    
    # Step 1: Determine if each index is a node or a leaf
    node_to_remove = []
    leafs_to_remove = []
    nodes = []
    leafs = []
    
    indices_to_find = indices_with_one + zero_sum_nodes
    
    for start_node in indices_to_find:
        all_nodes, leaf_nodes = get_subtree_info(start_node, left_child, right_child)
        leafs_to_remove.append(leaf_nodes)
        # print(f"Leaf nodes in the subtree rooted at node {start_node}: {leaf_nodes}")
        # Filter out the leaf nodes from all_nodes to get non-leaf nodes
        non_leaf_nodes = [node for node in all_nodes if node not in leaf_nodes]
        node_to_remove.append(non_leaf_nodes)
        

    
 

    
    node_to_remove = list(set(node for sublist in node_to_remove for node in sublist))
    leafs_to_remove = list(set(lfs for sublist in leafs_to_remove for lfs in sublist))
    
    for lf in leafs_to_remove:
        if (lf in right_child) and (lf-1 not in node_to_remove and lf-1 not in Leaf_nodes_pr):
            # print("raise issue-it should not be in the remove list")
            # print(lf)
            leafs_to_remove.remove(lf)
        
        if (lf in left_child) and (lf+1 not in node_to_remove and lf+1 not in Leaf_nodes_pr):
            # print("raise issue-it should not be in the remove list")
            # print(lf)
            leafs_to_remove.remove(lf)
        
        
        
    
    print("nodes to remove: ", node_to_remove)
    print("leafs to remove: ", leafs_to_remove)
    
    updated_left_child = copy.deepcopy(left_child)  # Deep copy
    updated_right_child = copy.deepcopy(right_child)  # Deep copy
    updated_feature_list = copy.deepcopy(feature_list)  # Deep copy
    updated_threshold_list = copy.deepcopy(threshold_list)  # Deep copy
    updated_values = copy.deepcopy(values)  # Deep copy

    #remove child nodes        
    for node in sorted (node_to_remove, reverse = True):
        updated_left_child[node] = -1
        updated_right_child[node] = -1
        updated_feature_list[node] = -2
        updated_threshold_list[node] = -2
        
    # Remove in reverse order to avoid index shifting
    for idx in sorted(leafs_to_remove, reverse=True):
        del updated_left_child[idx]
        del updated_right_child[idx]
        del updated_feature_list[idx] 
        del updated_threshold_list[idx]
        del updated_values[idx]
        
    biggest_number = max(updated_left_child + updated_right_child)
    print(biggest_number)
        
    return (updated_feature_list[:(len(updated_left_child)-1)], 
            updated_threshold_list[:(len(updated_left_child)-1)],
            updated_left_child, 
            updated_right_child, 
            updated_values[:len(updated_left_child)-1])





def renumber_tree_bfs(old_left, old_right, root=0):
    # Mapping from old IDs to new IDs
    mapping = {}
    new_left = []
    new_right = []

    # Queues for BFS
    queue = deque([root])
    next_new_id = 0

    # Initialize children arrays with a sufficient size guess; they will expand as needed.
    # Alternatively, we could dynamically append.
    new_left = []
    new_right = []

    while queue:
        old_id = queue.popleft()

        # If node is invalid, skip
        if old_id == -1:
            continue

        # Assign new ID for this node
        new_id = next_new_id
        next_new_id += 1
        mapping[old_id] = new_id

        # Ensure our new_left and new_right lists can accommodate the new_id.
        if new_id >= len(new_left):
            new_left.append(-1)
            new_right.append(-1)

        # Enqueue children for BFS before mapping them
        left_child_old = old_left[old_id] if old_id < len(old_left) else -1
        right_child_old = old_right[old_id] if old_id < len(old_right) else -1

        # We'll set the new children pointers after assigning new IDs to them.
        # For now, just record the old children in the queue
        queue.append(left_child_old)
        queue.append(right_child_old)

        # Reserve space for left/right child pointers; will fill after their IDs are assigned.
        new_left[new_id] = None  # placeholder
        new_right[new_id] = None  # placeholder

    # At this point, all nodes have been assigned new IDs in a level-order fashion.
    # Now create arrays for left/right children with proper indices.
    # We'll iterate over the mapping to set correct children for each new ID.
    for old_id, new_id in mapping.items():
        # For each original node, get its children
        left_old = old_left[old_id] if old_id < len(old_left) else -1
        right_old = old_right[old_id] if old_id < len(old_right) else -1

        # Map children to new IDs if they exist, otherwise -1
        new_left_child = mapping[left_old] if left_old in mapping else -1
        new_right_child = mapping[right_old] if right_old in mapping else -1

        # Set them in the new arrays
        new_left[new_id] = new_left_child
        new_right[new_id] = new_right_child

    return new_left, new_right, mapping

class SMCWrapperTree(BaseEstimator, ClassifierMixin):
    def __init__(self, features, thresholds, children_left, children_right, values, unique_labels):
        self.features = features
        self.thresholds = thresholds
        self.children_left = children_left
        self.children_right = children_right
        self.values = values  # Probability distributions at nodes
        self.unique_labels = unique_labels  # Original labels
    
    def fit(self, X, y=None):
        # Fit does not train, as the tree is already constructed
        self.n_classes_ = len(self.values[0][0])  # Number of classes
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        # Ensure the model is fitted
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("The model must be fitted before predicting.")

        predictions = []
        for sample in X:
            node_id = 0  # Start at the root
            while self.children_left[node_id] != -1:  # Traverse until a leaf
                feature = self.features[node_id]
                threshold = self.thresholds[node_id]
                if sample[feature] <= threshold:
                    node_id = self.children_left[node_id]
                else:
                    node_id = self.children_right[node_id]
            # Predict the class with the highest probability
            predicted_index = np.argmax(self.values[node_id])
            predictions.append(self.unique_labels[predicted_index])  # Use original labels
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Ensure the model is fitted
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("The model must be fitted before predicting probabilities.")

        probabilities = []
        for sample in X:
            node_id = 0  # Start at the root
            while self.children_left[node_id] != -1:  # Traverse until a leaf
                feature = self.features[node_id]
                threshold = self.thresholds[node_id]
                if sample[feature] <= threshold:
                    node_id = self.children_left[node_id]
                else:
                    node_id = self.children_right[node_id]
            # Append the probabilities at the leaf node
            probabilities.append(self.values[node_id][0])
        return np.array(probabilities)







import numpy as np
import logging
from art.classifiers.scikitlearn import ScikitlearnClassifier, ScikitlearnDecisionTreeClassifier

logger = logging.getLogger(__name__)

class FakeScikitlearnDecisionTreeClassifier(
    ScikitlearnDecisionTreeClassifier,  # so isinstance() passes
    ScikitlearnClassifier
):

    """
    A custom classifier that *pretends* to be a `ScikitlearnDecisionTreeClassifier`
    so that ART's `DecisionTreeAttack` will work. We bypass the usual type checks
    and implement the special methods the attack requires.
    """

    def __init__(self, custom_tree, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        :param custom_tree: An instance of your SMCWrapperTree (or similar).
        """
        # We intentionally call the *grandparent* constructor directly
        # to avoid the type-check in ScikitlearnDecisionTreeClassifier.__init__.
        # That check only accepts real sklearn.tree.DecisionTreeClassifier.
        # We want to skip that.
        ScikitlearnClassifier.__init__(
            self,
            model=custom_tree,               # pass your custom tree as "model"
            clip_values=clip_values,
            defences=defences,
            preprocessing=preprocessing
        )
        # Store our custom tree explicitly
        self._model = custom_tree

        # We can try to infer the shape/classes from the tree:
        self._input_shape = self._get_input_shape(custom_tree)
        self._nb_classes = self._get_nb_classes()

    ###################################################################
    # The following methods are the "decision tree" APIs used by ART’s
    # DecisionTreeAttack. They do NOT exist in the parent ScikitlearnClassifier,
    # so we implement them ourselves, delegating to SMCWrapperTree.
    ###################################################################

    def get_left_child(self, node_id):
        return self._model.children_left[node_id]

    def get_right_child(self, node_id):
        return self._model.children_right[node_id]

    def get_threshold_at_node(self, node_id):
        return self._model.thresholds[node_id]

    def get_feature_at_node(self, node_id):
        return self._model.features[node_id]

    def get_classes_at_node(self, node_id):
        """
        Return the major class (index) at this node.
        The original ART code calls np.argmax(tree_.value[node_id]).
        Here, `self._model.values[node_id]` might be shape (1, n_classes), or (n_classes,).
        Adjust as needed.
        """
        node_values = self._model.values[node_id]
        # If shape is (1, n_classes), flatten first:
        if len(node_values.shape) > 1:
            node_values = node_values[0]
        return np.argmax(node_values)

    def get_decision_path(self, x):
        """
        Return a subscriptable array of node indices from root to leaf
        for the first sample in `x` (or each sample if you wish).
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        sample = x[0]
        path = [0]  # Start at root node
        node_id = 0
        while self._model.children_left[node_id] != -1:
            feat = self._model.features[node_id]
            thr = self._model.thresholds[node_id]
            if sample[feat] <= thr:
                node_id = self._model.children_left[node_id]
            else:
                node_id = self._model.children_right[node_id]
            path.append(node_id)
            
        # Return a NumPy array that can be indexed with path[-1], path[-2], etc.
        return np.array(path, dtype=np.int32)


    ###################################################################
    # Optionally override some internal methods if needed
    ###################################################################

    def _get_input_shape(self, model):
        # If you can figure out the number of features from your tree, do so here.
        # For example:
        #   n_features = max(self._model.features) + 1
        # or store it in the tree during fit.
        if len(model.features) > 0:
            n_features = int(np.max(model.features)) + 1
            return (n_features,)
        else:
            logger.warning("No features found; setting input shape to None.")
            return None

    def _get_nb_classes(self):
        # Return how many classes the tree supports
        return len(self._model.unique_labels)

    def nb_classes(self):
        # Parent calls this to see how many classes exist
        return self._nb_classes



'''
trying to prune unneccesary nodes. 
not very succesfully done. WIth later version this wont be useful anw
pruned_feature_list, pruned_threshold_list, pruned_left_child, pruned_right_child, pruned_values = prune_tree_with_purity(
    feature_list, threshold_list, left_child, right_child, values, Leaf_nodes_pr
)

new_left_child, new_right_child, id_mapping = renumber_tree_bfs(pruned_left_child, pruned_right_child, root=0)

values, unique_labels_pruned = compute_values(
    X_train, y_train, 
    pruned_feature_list, pruned_threshold_list, new_left_child, new_right_child
)
'''

