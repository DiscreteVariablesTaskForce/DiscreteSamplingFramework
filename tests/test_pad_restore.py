import pytest
import numpy as np
from sklearn import datasets
from discretesampling.domain.decision_tree import Tree
from discretesampling.base.util import pad, restore


def build_tree(tree, leafs):
    data = datasets.load_wine()

    X = data.data
    y = data.target

    return Tree(X, y, tree, leafs)


@pytest.mark.parametrize(
    "particles,expected",
    [(  # two particles of same dim
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])],
        np.array([[6, 2, -1, 0, 1, 2, 2, 2.3, 0, 1, 2], [6, 2, -1, 0, 1, 2, 3, 15.2, 0, 1, 2]])
    ),
        (  # two particles of different dims
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])],
        np.array([[6, 2, -1, 0, 1, 2, 2, 2.3, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1],
                 [12, 3, -1, 0, 1, 2, 3, 15.2, 0, 2, 3, 4, 8, 1.64, 1, 1, 3, 4]])
    )]
)
def test_pad(particles, expected):
    padded_particles = pad(particles)
    assert all((padded_particles == expected).flatten())


@pytest.mark.parametrize(
    "padded_particles,expected",
    [(  # two particles of same dim
        np.array([[6, 2, -1, 0, 1, 2, 2, 2.3, 0, 1, 2], [6, 2, -1, 0, 1, 2, 3, 15.2, 0, 1, 2]]),
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])]
    ),
        (  # two particles of different dims
        np.array([[6, 2, -1, 0, 1, 2, 2, 2.3, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1],
                 [12, 3, -1, 0, 1, 2, 3, 15.2, 0, 2, 3, 4, 8, 1.64, 1, 1, 3, 4]]),
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])]
    )]
)
def test_restore(padded_particles, expected):
    particles = restore(padded_particles, [build_tree([0, 1, 2, 1, 1], [1, 2])])
    assert particles == expected


@pytest.mark.parametrize(
    "particles",
    [(  # two particles of same dim
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])]
    ),
        (  # two particles of different dims
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])]
    )]
)
def test_pad_restore(particles):
    padded_particles = pad(particles)
    new_particles = restore(padded_particles, [build_tree([0, 1, 2, 1, 1], [1, 2])])
    assert particles == new_particles
