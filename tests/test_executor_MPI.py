import pytest
import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.executor.executor_MPI import Executor_MPI


class ExampleParticleClass(DiscreteVariable):
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x


def split_across_cores(N, P, rank):
    local_n = int(np.ceil(N/P))
    first = rank*local_n
    last = np.min([first+local_n-1, N-1])
    return first, last


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5]), 5),
     (np.array([1, 1, 1, 1, 1]), 1),
     (np.array([0.0, 0.0, 1.23]), 1.23)]
)
def test_executor_max(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    max_x = exec.max(local_x)
    assert expected == max_x


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5]), 15),
     (np.array([1, 1, 1, 1, 1]), 5),
     (np.array([0.0, 0.0, 1.23]), 1.23)]
)
def test_executor_sum(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    sum_x = exec.sum(local_x)
    assert expected == sum_x


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([np.log(1.0)]), 0.0),  # single weight
     (np.array([np.log(0.5), np.log(0.5)]), 0.0),  # two even weights, 0.5,0.5
     (np.array([np.log(1.0), -np.inf]), 0.0),  # two weights, 1,0
     (np.array([np.log(1e5), np.log(1e-5), -np.inf]), 11.512925465070229)]
)
def test_logsumexp(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    calc = exec.logsumexp(local_x)
    np.testing.assert_almost_equal(calc, expected)  # use almost_equal for numerical inaccuracy


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5]), np.array([1, 3, 6, 10, 15])),
     (np.array([-1, -2, -3, -4, -5]), np.array([-1, -3, -6, -10, -15]))]
)
def test_cumsum(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    calc = exec.cumsum(local_x)
    assert all(calc == expected)


@pytest.mark.parametrize(
    "particles,ncopies,expected",
    [([ExampleParticleClass(x) for x in ["a", "b", "c", "d", "e"]], np.array(
        [0, 2, 1, 0, 2]), [ExampleParticleClass(x) for x in ["b", "b", "c", "e", "e"]])]
)
def test_redistribute(particles, ncopies, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(particles), exec.P, exec.rank))
    local_particles = particles[first:(last+1)]
    local_ncopies = ncopies[first:(last+1)]
    local_new_particles = exec.redistribute(local_particles, local_ncopies)
    new_particles = exec.gather(local_new_particles, )
    assert new_particles == expected
