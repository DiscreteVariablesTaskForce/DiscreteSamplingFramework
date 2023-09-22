import pytest
import numpy as np
from sklearn import datasets
from discretesampling.domain.decision_tree import Tree
from discretesampling.base.executor.executor import Executor
from discretesampling.base.executor.MPI.variable_size_redistribution import variable_size_redistribution


def build_tree(tree, leafs):
    data = datasets.load_wine()

    X = data.data
    y = data.target

    return Tree(X, y, tree, leafs)


@pytest.mark.parametrize(
    "particles,ncopies,expected",
    [(  # two particles of same dim, equal num copies
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 1])],
        np.array([1, 1]),
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 1])]
    ),
        (  # two particles of different dims, equal num copies
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])],
        np.array([1, 1]),
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])]
    ),
        (  # two particles of different dims, unequal num copies
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])],
        np.array([2, 0]),
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2])]
    ),
        (  # two particles of different dims, equal num copies
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])],
        np.array([0, 2]),
        [build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4]),
         build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])]
    )]
)
def test_variable_size_redistribution(particles, ncopies, expected):
    new_particles = variable_size_redistribution(particles, ncopies, exec=Executor())
    assert new_particles == expected
