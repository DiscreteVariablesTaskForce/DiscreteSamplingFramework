import pytest
import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.executor.executor_MPI import Executor_MPI


class ExampleParticleClass(DiscreteVariable):
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x

    def getProposalType(self):
        return super().getProposalType()

    def getTargetType(self):
        return super().getTargetType()


def split_across_cores(N, P, rank):
    local_n = int(np.ceil(N/P))
    first = rank*local_n
    last = np.min([first+local_n-1, N-1])
    return first, last


@pytest.mark.mpi
@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 8),
     (np.array([1, 1, 1, 1, 1, 1, 1, 1]), 1),
     (np.array([0.0, 0.0, 1.23, 0.0, 0.0, 0.0, 0.0]), 1.23)]
)
def test_executor_max(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    max_x = exec.max(local_x)
    assert expected == max_x


@pytest.mark.mpi
@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 36.0),
     (np.array([1, 1, 1, 1, 1, 1, 1, 1]), 8),
     (np.array([0.0, 0.0, 1.23, 0.0, 0.0, 0.0, 0.0, 0.0]), 1.23)]
)
def test_executor_sum(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    sum_x = exec.sum(local_x)
    assert expected == sum_x


@pytest.mark.mpi
@pytest.mark.parametrize(
    "x",  # Only functions for equal numbers per core
    [(np.array([1, 2, 3, 4, 5, 6, 7, 8])),
     (np.array([1, 1, 1, 1, 1, 1, 1, 1])),
     (np.array([0.0, 0.0, 1.23, 4.56, 7.89, 0.0, 1.23, 0.0]))]
)
def test_executor_gather(x):
    all_x_shape = x.shape
    exec = Executor_MPI()
    first, last = split_across_cores(len(x), exec.P, exec.rank)
    local_x = x[first:(last+1)]
    all_x = exec.gather(local_x, all_x_shape)
    assert all(x == all_x)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "x,expected",
    [(np.array([-np.log(8)]*8), 0.0),  # eight even weights
     (np.array([np.log(1.0), -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]), 0.0),  # 8 weights, 1,0,...,0
     (np.array([np.log(1e5), np.log(1e-5), -np.inf, np.log(1e10), 0.0, np.log(1e-2), -np.inf, 0.0]), 23.025860930091458)]
)
def test_logsumexp(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    calc = exec.logsumexp(local_x)
    np.testing.assert_almost_equal(calc, expected)  # use almost_equal for numerical inaccuracy


@pytest.mark.mpi
@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5, 6, 7, 8]), np.array([1, 3, 6, 10, 15, 21, 28, 36])),
     (np.array([-1, -2, -3, -4, -5, -6, -7, -8]), np.array([-1, -3, -6, -10, -15, -21, -28, -36]))]
)
def test_cumsum(x, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(x), exec.P, exec.rank))
    local_x = x[first:(last+1)]
    calc = exec.cumsum(local_x)
    local_expected = expected[first:(last+1)]
    assert all(calc == local_expected)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "particles,ncopies,expected",  # Only functions for equal numbers per core
    [([ExampleParticleClass(x) for x in ["a", "b", "c", "d", "e", "f", "g", "h"]], np.array(
        [0, 2, 1, 0, 2, 1, 0, 2]), [ExampleParticleClass(x) for x in ["b", "b", "c", "e", "e", "f", "h", "h"]])]
)
def test_redistribute(particles, ncopies, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(particles), exec.P, exec.rank))
    local_particles = particles[first:(last+1)]
    local_ncopies = ncopies[first:(last+1)]
    local_new_particles = exec.redistribute(local_particles, local_ncopies)
    local_expected = expected[first:(last+1)]
    assert local_new_particles == local_expected
