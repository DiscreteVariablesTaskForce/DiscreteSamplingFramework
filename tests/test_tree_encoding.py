import pytest
import numpy as np
from sklearn import datasets
from discretesampling.domain.decision_tree import Tree
from discretesampling.domain.decision_tree.util import encode_move, decode_move, extract_tree, extract_leafs


def build_tree(tree, leafs):
    data = datasets.load_wine()

    X = data.data
    y = data.target

    return Tree(X, y, tree, leafs)


@pytest.mark.parametrize(
    "tree,expected",
    [(build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), np.array([6, 2, -1, 0, 1, 2, 2, 2.3, 0, 1, 2])),
     (build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2]), np.array([6, 2, -1, 0, 1, 2, 3, 15.2, 0, 1, 2])),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      np.array([12, 3, -1, 0, 1, 2, 3, 15.2, 0, 2, 3, 4, 8, 1.64, 1, 1, 3, 4.]))]
)
def test_encode_tree(tree, expected):
    encoded_tree = tree.encode(tree)
    assert all(encoded_tree == expected)


@pytest.mark.parametrize(
    "move,expected",
    [("grow", 0),
     ("prune", 1),
     ("swap", 2),
     ("change", 3),
     ("", -1)]
)
def test_encode_move(move, expected):
    encoded_move = encode_move(move)
    assert encoded_move == expected


@pytest.mark.parametrize(
    "encoded_tree,expected",
    [(np.array([0, 1, 2, 2, 2.3, 0]), [[0, 1, 2, 2, 2.3, 0]]),
     (np.array([0, 1, 2, 3, 15.2, 0]), [[0, 1, 2, 3, 15.2, 0]]),
     (np.array([0, 1, 2, 3, 15.2, 0, 2, 3, 4, 8, 1.64, 1]), [[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]])]
)
def test_extract_tree(encoded_tree, expected):
    extracted_tree = extract_tree(encoded_tree)
    assert extracted_tree == expected


@pytest.mark.parametrize(
    "encoded_tree,expected",
    [(np.array([1., 2.]), [1, 2]),
     (np.array([1., 2.]), [1, 2]),
     (np.array([1., 3., 4.]), [1, 3, 4])]
)
def test_extract_leafs(encoded_tree, expected):
    extracted_leafs = extract_leafs(encoded_tree)
    assert extracted_leafs == expected


@pytest.mark.parametrize(
    "code,expected",
    [(0, "grow"),
     (1, "prune"),
     (2, "swap"),
     (3, "change"),
     (-1, "")]
)
def test_decode_move(code, expected):
    decoded_move = decode_move(code)
    assert decoded_move == expected
