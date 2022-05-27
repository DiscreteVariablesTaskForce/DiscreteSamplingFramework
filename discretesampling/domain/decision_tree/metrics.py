import math
import numpy as np
import collections
from .util import find_leaf_for_datum


class stats():
    def __init__(self, tree, X_test):
        self.X_train = tree.X_train
        self.y_train = tree.y_train
        self.tree = tree

    def predict(self, X_test):
        leaf_possibilities = getLeafPossibilities(self.tree)
        leafs = sorted(self.tree.leafs)
        labels = []
        for datum in X_test:
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            flag = "false"
            current_node = self.tree.tree[0]
            label_max = -1
            # make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in leafs and flag == "false":
                if datum[current_node[3]] > current_node[4]:
                    for node in self.tree.tree:
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

                            labels.append(actual_label)
                            break

                else:
                    for node in self.tree.tree:
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

                            labels.append(actual_label)
                            break

            if current_node in leafs:
                indice = current_node.index(current_node)
                # find in the dictionary which is the highest probable label
                for x, y in leaf_possibilities[indice].items():
                    if y == label_max:
                        actual_label = 1

                    if y > label_max:
                        label_max = y
                        actual_label = x
        return (labels)


def accuracy(y_test, labels):
    correct_classification = 0
    for i in range(len(y_test)):
        if labels[i] == y_test[i]:
            correct_classification += 1

    acc = correct_classification*100/len(y_test)
    return acc
    correct_classification = 0
    for i in range(len(y_test)):
        if labels[i] == y_test[i]:
            correct_classification += 1

        acc = correct_classification*100/len(y_test)
        return acc


def getLeafPossibilities(x):
    target1, leafs_possibilities_for_prediction = calculate_leaf_occurences(x)
    return leafs_possibilities_for_prediction


# Π(Y_i|T,theta,x_i)
def calculate_leaf_occurences(x):
    '''
    we calculate how many labelled as 0 each leaf has, how many labelled as 1
    each leaf has and so on
    '''
    leafs = x.leafs
    leafs_possibilities = [collections.Counter() for _ in leafs]
    for k, datum in enumerate(x.X_train):

        leaf = find_leaf_for_datum(x, datum)
        leafs_possibilities[x.leafs.index(leaf)][x.y_train[k]] += 1

    unique = list(set(x.y_train))

    for item in leafs_possibilities:
        if len(item) == 0:
            for i in unique:
                item[i] = 1

    # create leafs possibilities
    for item in leafs_possibilities:
        factor = 1.0/sum(item.values())
        for k in item:
            item[k] = item[k]*factor

    product_of_leafs_probabilities = []
    for k, datum in enumerate(x.X_train):
        leaf = find_leaf_for_datum(x, datum)

        probs = leafs_possibilities[leafs.index(leaf)]
        target_probability = probs[x.y_train[k]]
        target_probability = max(0.02, min(0.98, target_probability))

        product_of_leafs_probabilities.append(math.log(target_probability))

    product_of_target_feature = np.sum(product_of_leafs_probabilities)
    return product_of_target_feature, leafs_possibilities
