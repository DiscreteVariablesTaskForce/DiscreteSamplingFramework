import math
import numpy as np
from collections import Counter
import pandas as pd


# MJAS fixed extremely confusing arg order!
class stats():
    def __init__(self, trees, X_test):
        self.trees = trees
        #self.X_train = X_train
        #self.y_train = y_train
        self.X_test = X_test
        #self.y_test = y_test

    def getLeafPossibilities(self, x):
        target1, leafs_possibilities_for_prediction = calculate_leaf_occurences(x)
        return leafs_possibilities_for_prediction

    def majority_voting_predict(self, smcLabels):  # this function should be moved to a more appropriate place
        labels = []
        predictions = pd.DataFrame(smcLabels)
        for column in predictions:
            labels.append(predictions[column].mode())
        labels = pd.DataFrame(labels)
        labels = labels.values.tolist()
        labels1 = []
        if len(labels[0]) > 1:
            for label in labels:
                labels1.append(label[0])
            # acc = dt.accuracy(y_test, labels1)
            labels = labels1
        # else:
            # acc = dt.accuracy(y_test, labels)
        labels = list(np.array(labels).flatten())
        return labels

    def predict(self, X_test, use_majority=True):
        all_labels_from_all_trees = []
        for tree in self.trees:
            all_labels_from_this_trees = self.predict_for_one_tree(tree, X_test)
            all_labels_from_all_trees.append(all_labels_from_this_trees)

        if use_majority:
            return self.majority_voting_predict(all_labels_from_all_trees)
        else:
            return all_labels_from_all_trees

    def predict_for_one_tree(self, tree, X_test):
        all_labels_for_this_tree = []
        leaf_possibilities = self.getLeafPossibilities(tree)
        leafs = sorted(tree.leafs)
        for datum in X_test:
            label_for_this_datum = self.predict_for_one_datum(tree, leafs, leaf_possibilities, datum)
            all_labels_for_this_tree.append(label_for_this_datum)
        return all_labels_for_this_tree

    def predict_for_one_datum(self, tree, leafs, leaf_possibilities, datum):

        label = None
        flag = "false"
        current_node = tree.tree[0]
        label_max = -1
        # make sure that we are not in leafs. current_node[0] is the node
        while current_node[0] not in leafs and flag == "false":
            if datum[current_node[3]] > current_node[4]:
                for node in tree.tree:
                    if node[0] == current_node[2]:
                        current_node = node
                        break
                    if current_node[2] in leafs:
                        leaf = current_node[2]
                        flag = "true"
                        indice = leafs.index(leaf)
                        for x, y in leaf_possibilities[indice].items():
                            if y == label_max:
                                actual_label = 1

                            if y > label_max:
                                label_max = y
                                actual_label = x

                        label = actual_label
                        break

            else:
                for node in tree.tree:
                    if node[0] == current_node[1]:
                        current_node = node
                        break
                    if current_node[1] in leafs:
                        leaf = current_node[1]
                        flag = "true"
                        indice = leafs.index(leaf)
                        for x, y in leaf_possibilities[indice].items():
                            if y == label_max:
                                actual_label = 1

                            if y > label_max:
                                label_max = y
                                actual_label = x

                        label = actual_label
                        break

        return label


def accuracy(y_test, labels):
    correct_classification = 0
    for i in range(len(y_test)):
        if labels[i] == y_test[i]:
            correct_classification += 1

    acc = correct_classification*100/len(y_test)
    return acc


# Π(Y_i|T,theta,x_i)
# MJAS rewritten for efficiency and making corrections for hacked probabilities
# ~30 times faster
# just get counts - calculate likelihood as post processing
# now uses the safe Dirichelet/Multinomial approach: p_j = (count_j + 1)/Sum_j(count_j +1)
def improved_leaf_occurences(x, X_train, y_train, current_node_id = None): # set current_node = x.tree[0]
    ### really need to index the tree!
    if current_node_id is None:
        current_node_id = x.tree[0][0]
    #print("received", len(X_train))
    if current_node_id in x.leafs:
        # do actual calcs
        #print("Process leaf", current_node_id)
        return [(current_node_id, Counter(y_train))]
    # otherwise .. use the threshold
    for current_node in x.tree:
        if current_node[0] == current_node_id:
            break
    #print("Process non-leaf", current_node)
    thresh = current_node[4]
    feat_num = current_node[3]
    mask = X_train[:,feat_num] > thresh
    occ = []
    nr = np.sum(mask)
    if nr < mask.shape[0]: # some go left
        #print(" left child id = ", current_node[1])
        occ_left = improved_leaf_occurences(x, X_train[~mask], y_train[~mask], current_node[1])
        occ += occ_left # concatenate lists (of counters)
    if nr > 0: # some go right
        #print(" right child id = ", current_node[2])
        occ_right = improved_leaf_occurences(x, X_train[mask], y_train[mask], current_node[2])
        occ += occ_right
    return(occ)

# get expected and actual occupency for each decision node and leaf (in case we want proposals that are local to a subset)
def node_occurences(x, X_train, expected = None, current_node_id = None):
    if expected is None:
        expected = float(X_train.shape[0])
    if current_node_id is None:
        current_node_id = x.tree[0][0]
    if current_node_id in x.leafs:
        return([(current_node_id, expected, len(X_train))])
    # find this node
    for current_node in x.tree:
        if current_node[0] == current_node_id:
            break
    thresh = current_node[4]
    feat_num = current_node[3]
    mask = X_train[:,feat_num] > thresh
    occ = [(current_node_id, expected, len(X_train))]
    occ_left = node_occurences(x, X_train[~mask], expected * 0.5, current_node[1])
    occ += occ_left # concatenate lists (of counters)
    occ_right = node_occurences(x, X_train[mask], expected * 0.5, current_node[2])
    occ += occ_right
    return(occ)


# MJAS efficient replacement for the existing function with the same name
# now returns either log likelihood or accuracy depending on flag
def calculate_leaf_occurences(x, X_test = None, y_test = None, pseudo_count = 1, categories = None, verbose = False, accuracy = False):
    if categories is None:
        categories = np.sort(np.unique(x.y_train)) # DANGER you should pass the categories in when working with subsets and pseudo count > 0
    occ_dict = dict(improved_leaf_occurences(x, x.X_train, x.y_train))
    # now get likelihood
    # we can have DIFFERENT test data
    if not X_test is None:
        occ_dict_test = dict(improved_leaf_occurences(x, x.X_test, x.y_test))
    else:
        occ_dict_test = occ_dict
    metric = 0.0    
    total_test_count = 0
    occ = []
    if verbose:
        print(x.leafs)
        print(occ_dict)
    for l in x.leafs:
        regularised_counts = Counter(dict([(c, pseudo_count + 1e-12) for c in categories]))
        if l in occ_dict.keys():
            regularised_counts.update(occ_dict[l])
        #test_counts = copy.deepcopy(regularised_counts)
        #test_counts.subtract(pseudo_counts) # original counts but with zeros where needed 
        rcval = np.array(list(regularised_counts.values()))
        denom = np.sum(rcval)
        probs = rcval / denom
        #
        if l in occ_dict_test.keys():
            test_counts = occ_dict_test[l]
        else:
            test_counts = regularised_counts
            test_counts.subtract(regularised_counts) # array of zeros
        cval = np.array([test_counts[c] for c in regularised_counts.keys()])
        total_test_count += np.sum(cval) # TO DO this only works if pseudo count > 0 so regularised counts is complete
        if accuracy:
            metric += np.sum(cval * probs)
        else:
            log_total_probs = cval * np.log(probs)
            metric += np.sum(log_total_probs)       
        #
        occ.append(Counter(dict(zip(regularised_counts.keys(),probs)))) # just for backward compat
    return((metric / total_test_count) if accuracy else metric, occ)