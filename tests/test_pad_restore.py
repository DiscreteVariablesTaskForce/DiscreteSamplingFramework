import pytest
import numpy as np
from sklearn import datasets
from discretesampling.domain.decision_tree import Tree
from discretesampling.base.util import pad, restore, gather_all
from discretesampling.base.executor.executor import Executor
from discretesampling.base.executor.executor_MPI import Executor_MPI


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
    padded_particles = pad(particles, exec=Executor())
    assert all((padded_particles == expected).flatten())


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
@pytest.mark.mpi
def test_pad_MPI(particles, expected):
    exec = Executor_MPI()
    particles = particles * exec.P  # copy particles to ensure there is more than one per rank
    N = len(particles)
    loc_n = int(N/exec.P)
    indexes = [i + exec.rank*loc_n for i in range(loc_n)]
    local_particles = [particles[i] for i in indexes]

    padded_particles = pad(local_particles, exec=exec)
    return_shape = [N, padded_particles.shape[1]]
    all_padded_particles = exec.gather(padded_particles, return_shape)
    expected = np.tile(expected, [exec.P, 1])  # tile expected result to match
    assert all((all_padded_particles == expected).flatten())


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
    padded_particles = pad(particles, exec=Executor())
    new_particles = restore(padded_particles, [build_tree([0, 1, 2, 1, 1], [1, 2])])
    assert particles == new_particles


@pytest.mark.parametrize(
    "particles",
    [(  # two particles of same dim
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0]], [1, 2])]
    ),
        (  # two particles of different dims
        [build_tree([[0, 1, 2, 2, 2.3, 0]], [1, 2]), build_tree([[0, 1, 2, 3, 15.2, 0], [2, 3, 4, 8, 1.64, 1]], [1, 3, 4])]
    )]
)
def test_gather_all(particles):
    new_particles = gather_all(particles, exec=Executor())
    assert particles == new_particles
