import pytest
from discretesampling.domain.decision_tree import calculate_leaf_occurences, Tree
from discretesampling.base.random import RandomInt
from collections import Counter
from sklearn import datasets


def data():
    data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


def build_tree(feature, threshold):
    return build_tree_k_depth(1, [feature], [threshold])


def build_tree_k_depth(k, features=None, thresholds=None):
    X, y = data()
    num_nodes = 2**k - 1
    if features is None:
        features = [RandomInt(0, X.shape[1] - 1).eval() for _ in range(num_nodes)]
    if thresholds is None:
        thresholds = [RandomInt(0, X.shape[0] - 1).eval() for _ in range(num_nodes)]

    leafs = range(num_nodes, num_nodes+2**k)
    nodeid = 0
    tree = []
    for d in range(k):
        for i in range(2**d):
            nodeid = 2**d - 1 + i
            node = [nodeid, 2**(d) + nodeid, 2**(d) + nodeid + 1, features[nodeid], X[thresholds[nodeid], features[nodeid]]]
            tree.append(node)

    return Tree(X, y, tree, leafs)


@pytest.mark.parametrize(
    "tree,expected",
    [(build_tree(2, 10), (-182.25009765866628, [Counter({1: 0.6000000000000001,
                                                         0: 0.22666666666666668,
                                                         2: 0.17333333333333334}),
                                                Counter({0: 0.4077669902912621,
                                                         2: 0.3398058252427184,
                                                         1: 0.2524271844660194})])),
     (build_tree(2, 20), (-182.469284951157, [Counter({1: 0.6212121212121212,
                                                       0: 0.21212121212121213,
                                                       2: 0.16666666666666669}),
                                              Counter({0: 0.40178571428571425,
                                                       2: 0.33035714285714285,
                                                       1: 0.26785714285714285})])),
     (build_tree(4, 10), (-184.44864692209964, [Counter({1: 0.4841269841269841,
                                                         2: 0.2698412698412698,
                                                         0: 0.24603174603174602}),
                                                Counter({0: 0.5384615384615385,
                                                         2: 0.2692307692307693,
                                                         1: 0.19230769230769232})])),
     (build_tree_k_depth(2), (-150.22738860813286, [Counter({1: 0.5, 2: 0.5}),
                                                    Counter({1: 0.5116279069767442,
                                                             2: 0.3488372093023256,
                                                             0: 0.13953488372093023}),
                                                    Counter({0: 0.8723404255319148,
                                                             1: 0.0851063829787234,
                                                             2: 0.0425531914893617}),
                                                    Counter({0: 0.3333333333333333,
                                                             1: 0.3333333333333333,
                                                             2: 0.3333333333333333})])),
     (build_tree_k_depth(3), (-148.9726091122371, [Counter({2: 0.5423728813559322,
                                                            1: 0.3898305084745763,
                                                            0: 0.06779661016949153}),
                                                   Counter({1: 0.547945205479452, 0: 0.4520547945205479}),
                                                   Counter({0: 0.4772727272727273, 2: 0.34090909090909094,
                                                            1: 0.18181818181818182}),
                                                   Counter({0: 0.5, 2: 0.5}),
                                                   Counter({0: 0.3333333333333333, 1: 0.3333333333333333,
                                                            2: 0.3333333333333333}),
                                                   Counter({0: 0.3333333333333333, 1: 0.3333333333333333,
                                                            2: 0.3333333333333333}),
                                                   Counter({0: 0.3333333333333333, 1: 0.3333333333333333,
                                                            2: 0.3333333333333333}),
                                                   Counter({0: 0.3333333333333333, 1: 0.3333333333333333,
                                                            2: 0.3333333333333333})]))]
)
def test_calculate_leaf_occurences(tree, expected):
    leafs_possibilities_for_prediction = calculate_leaf_occurences(tree)
    assert leafs_possibilities_for_prediction == expected
