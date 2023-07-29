import pytest
import numpy as np
from sklearn import datasets
from discretesampling.domain.decision_tree import Tree
from discretesampling.base.kernel import DiscreteVariableOptimalLKernel


def getData():
    data = datasets.load_wine()

    X = data.data
    y = data.target
    return (X, y)


def build_tree(tree, leafs):
    X, y = getData()
    return Tree(X, y, tree, leafs)


@pytest.mark.parametrize(
    "current_particles,previous_particles, index, expected",
    [(  # two particles of same dim
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])],
        [build_tree([[0, 1, 2, 2, 15.2, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])],
        0, -7.05444965813294
    )]
)
def test_optimal_L_eval(current_particles, previous_particles, index, expected):
    optL = DiscreteVariableOptimalLKernel(current_particles, previous_particles)
    logprob = optL.eval(index)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.skip
@pytest.mark.parametrize(
    "current_particles,previous_particles, index, expected",
    [(  # two particles of same dim
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])],
        [build_tree([[0, 1, 2, 2, 15.2, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])],
        0, -7.05444965813294
    )]
)
def test_parallel_optimal_L_eval(current_particles, previous_particles, index, expected):
    optL = DiscreteVariableOptimalLKernel(
        current_particles, previous_particles, parallel=True, num_cores=2
    )
    logprob = optL.eval(index)
    np.testing.assert_almost_equal(logprob, expected)
