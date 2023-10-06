import pytest
import numpy as np
from sklearn import datasets
from copy import copy
from discretesampling.base.random import RNG
from discretesampling.domain.decision_tree import (
    Tree, TreeProposal, TreeInitialProposal, TreeTarget
)
from discretesampling.domain.decision_tree.util import encode_move, decode_move, extract_tree, extract_leafs


def getData():
    data = datasets.load_wine()

    X = data.data
    y = data.target
    return (X, y)


def build_tree(tree, leafs):
    X, y = getData()
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


@pytest.mark.parametrize(
    "tree,leaf,expected",
    [(build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]), 1, 1),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]), 3, 2),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]), 4, 2)]
)
def test_tree_depth_of_leaf(tree, leaf, expected):
    depth = tree.depth_of_leaf(leaf)
    assert depth == expected


@pytest.mark.parametrize(
    "seed,tree,index,expected",
    [(0, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      0, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [1, 5, 6, 11, 2.31, 1]], [3, 4, 5, 6])),
     (1, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      1, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [3, 5, 6, 6, 1.25, 2]], [1, 4, 5, 6])),
     (2, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      2, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 10, 1.04, 2]], [1, 3, 5, 6]))]
)
def test_tree_grow_leaf(seed, tree, index, expected):
    new_tree = copy(tree)
    new_tree.grow_leaf(index, rng=RNG(seed))
    assert new_tree == expected


@pytest.mark.parametrize(
    "seed,tree,expected",
    [(0, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 8, 1.64, 2]], [1, 3, 5, 6])),
     (1, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [3, 5, 6, 6, 0.58, 2]], [1, 4, 5, 6])),
     (2, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6]))]
)
def test_tree_grow(seed, tree, expected):
    tree.grow(rng=RNG(seed))
    assert tree == expected


@pytest.mark.parametrize(
    "seed,tree,expected",
    [(0, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])),
     (1, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])),
     (2, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])),
     (3, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])),
     (4, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2]))]
)
def test_tree_prune(seed, tree, expected):
    tree.prune(rng=RNG(seed))
    assert tree == expected


@pytest.mark.parametrize(
    "seed,tree,expected",
    [(0, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])),
     (1, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 6, 0.58, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])),
     (2, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 3, 15.2, 1]], [1, 3, 4]))]
)
def test_tree_change(seed, tree, expected):
    tree.change(rng=RNG(seed))
    assert tree == expected


@pytest.mark.parametrize(
    "seed,tree,expected",
    [(0, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 3, 15.2, 1], [4, 5, 6, 8, 1.64, 2]], [1, 3, 5, 6])),
     (1, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 3, 15.2, 1], [4, 5, 6, 8, 1.64, 2]], [1, 3, 5, 6])),
     (2, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1], [4, 5, 6, 3, 15.2, 2]], [1, 3, 5, 6])),
     (3, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 8, 1.64, 0], [2, 3, 4, 3, 15.2, 1]], [1, 3, 4])),
     (4, build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
      build_tree([[0, 1, 2, 8, 1.64, 0], [2, 3, 4, 3, 15.2, 1]], [1, 3, 4]))]
)
def test_tree_swap(seed, tree, expected):
    tree.swap(rng=RNG(seed))
    assert tree == expected


@pytest.mark.parametrize(
    "seed,tree,expected",
    [(0, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 3, 16.4, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6])),
     (1, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 12, 1290.0, 2]], [3, 4, 5, 6])),
     (2, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 8, 1.64, 0], [1, 3, 4, 3, 15.2, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6])),
     (3, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1]], [2, 3, 4])),
     (4, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1],
                  [2, 5, 6, 3, 15.2, 2], [6, 7, 8, 6, 0.66, 3]], [3, 4, 5, 7, 8]))]
)
def test_tree_proposal_sample(seed, tree, expected):
    new_tree = TreeProposal().sample(tree, rng=RNG(seed))
    assert new_tree == expected


@pytest.mark.parametrize(
    "tree_a,tree_b,expected",
    [(build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 3, 16.4, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      -9.643852892639503),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 12, 1290.0, 2]], [3, 4, 5, 6]),
      -9.643852892639503),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 8, 1.64, 0], [1, 3, 4, 3, 15.2, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      -3.401197381662155),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1]], [2, 3, 4]),
      -2.0794415416798357),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [
          2, 5, 6, 3, 15.2, 2], [6, 7, 8, 6, 0.66, 3]], [3, 4, 5, 7, 8]),
      -10.519321629993403)]
)
def test_tree_proposal_eval(tree_a, tree_b, expected):
    logprob = TreeProposal().eval(tree_a, tree_b)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "seed,expected",
    [(0, build_tree([[0, 1, 2, 11, 2.31, 0]], [1, 2])),
     (1, build_tree([[0, 1, 2, 6, 1.25, 0]], [1, 2])),
     (2, build_tree([[0, 1, 2, 10, 1.04, 0]], [1, 2])),
     (3, build_tree([[0, 1, 2, 10, 1.28, 0]], [1, 2])),
     (4, build_tree([[0, 1, 2, 9, 10.26, 0]], [1, 2]))]
)
def test_tree_initial_proposal_sample(seed, expected):
    X, y = getData()
    tree = TreeInitialProposal(X, y).sample(rng=RNG(seed))
    assert tree == expected


@pytest.mark.parametrize(
    "tree,expected",
    [(build_tree([[0, 1, 2, 11, 2.31, 0]], [1, 2]),
     -7.746732907753621),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 3, 16.4, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
     -7.746732907753621),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 12, 1290.0, 2]], [3, 4, 5, 6]),
     -7.746732907753621),
     (build_tree([[0, 1, 2, 8, 1.64, 0], [1, 3, 4, 3, 15.2, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
     -7.746732907753621),
     (build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [
         2, 5, 6, 3, 15.2, 2], [6, 7, 8, 6, 0.66, 3]], [3, 4, 5, 7, 8]),
     -7.746732907753621)]
)
def test_tree_initial_proposal_eval(tree, expected):
    X, y = getData()
    logprob = TreeInitialProposal(X, y).eval(tree)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "seed,a,b,expected",  # Addition of 'target' doesn't appear to affect sampling?
    [(0, 0.01, None, build_tree([[0, 1, 2, 11, 2.31, 0]], [1, 2])),
     (1, 0.01, None, build_tree([[0, 1, 2, 6, 1.25, 0]], [1, 2])),
     (2, 1,   None, build_tree([[0, 1, 2, 10, 1.04, 0]], [1, 2])),
     (3, 5,   None, build_tree([[0, 1, 2, 10, 1.28, 0]], [1, 2])),
     (4, 10,  None, build_tree([[0, 1, 2, 9, 10.26, 0]], [1, 2]))]
)
def test_tree_initial_proposal_sample_target(seed, a, b, expected):
    X, y = getData()
    target = TreeTarget(a, b)
    tree = TreeInitialProposal(X, y).sample(rng=RNG(seed), target=target)
    assert tree == expected


@pytest.mark.parametrize(
    "a,b,tree,expected",
    [(0.01, None, build_tree([[0, 1, 2, 11, 2.31, 0]], [1, 2]),
      -13.050054440964843),
     (0.01, None, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 3, 16.4, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      -24.745301462729024),
     (1, None, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 12, 1290.0, 2]], [3, 4, 5, 6]),
      -11.466111592714485),
     (5, None, build_tree([[0, 1, 2, 8, 1.64, 0], [1, 3, 4, 3, 15.2, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      -9.480274338915677),
     (10, None, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [
         2, 5, 6, 3, 15.2, 2], [6, 7, 8, 6, 0.66, 3]], [3, 4, 5, 7, 8]),
      -11.021253784605069)]
)
def test_tree_initial_proposal_eval_target(a, b, tree, expected):
    X, y = getData()
    target = TreeTarget(a, b)
    logprob = TreeInitialProposal(X, y).eval(tree, target=target)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "a,b,tree,expected",
    [(0.01, None, build_tree([[0, 1, 2, 11, 2.31, 0]], [1, 2]),
      -129.26086380987735),
     (0.01, None, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 3, 16.4, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      -218.67962673743085),
     (1, None, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [2, 5, 6, 12, 1290.0, 2]], [3, 4, 5, 6]),
      -197.2117234957705),
     (5, None, build_tree([[0, 1, 2, 8, 1.64, 0], [1, 3, 4, 3, 15.2, 1], [2, 5, 6, 3, 15.2, 2]], [3, 4, 5, 6]),
      -177.30847255069276),
     (10, None, build_tree([[0, 1, 2, 3, 15.2, 0], [1, 3, 4, 8, 1.64, 1], [
         2, 5, 6, 3, 15.2, 2], [6, 7, 8, 6, 0.66, 3]], [3, 4, 5, 7, 8]),
      -175.6848813061998)]
)
def test_tree_target_eval(a, b, tree, expected):
    logprob = TreeTarget(a, b).eval(tree)
    np.testing.assert_almost_equal(logprob, expected)
