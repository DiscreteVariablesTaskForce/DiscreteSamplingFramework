import pytest
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base.executor.executor import Executor
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling, check_stability


def split_across_cores(N, P, rank):
    local_n = int(np.ceil(N/P))
    first = rank*local_n
    last = np.min([first+local_n-1, N-1])
    return first, last


@pytest.mark.parametrize(
    "logw,expected_ess",
    [(np.array([np.log(1.0)]), 1.0),  # single weight
     (np.array([np.log(0.5), np.log(0.5)]), 2.0),  # two even weights, 0.5,0.5
     (np.array([np.log(1.0), -np.inf]), 1.0)]  # two weights, 1,0
)
def test_ess(logw, expected_ess):
    calc_ess = ess(logw, exec=Executor())
    np.testing.assert_allclose(calc_ess, expected_ess)  # use almost_equal for numerical inaccuracy


@pytest.mark.parametrize(
    "array,expected",
    [(np.array([0.0, 0.0]), np.array([np.log(0.5), np.log(0.5)])),  # equal, non-normalised weights
     (np.array([np.log(0.25), np.log(0.75)]), np.array([np.log(0.25), np.log(0.75)]))  # already normalised
     ]
)
def test_log_normalise(array, expected):
    normalised_array = normalise(array, exec=Executor())
    np.testing.assert_allclose(normalised_array, expected)  # use allclose for numerical inaccuracy


@pytest.mark.parametrize(
    "particles,logw,expected",

    [([1, 2, 3], np.array([-np.log(3), -np.log(3), -np.log(3)]), [1, 2, 3]),
     ([1, 2, 3], np.array([np.log(1), -np.inf, -np.inf]), [1, 1, 1]),
     ([1, 2, 3], np.array([np.log(0.5), np.log(0.5), -np.inf]), [1, 2, 2]),
     ([1, 2, 3, 4], np.array([np.log(0.5), -np.inf, np.log(0.5), -np.inf]), [1, 1, 3, 3])]
)
def test_systematic_resampling(particles, logw, expected):
    N = len(particles)
    new_particles, new_logw = systematic_resampling(particles, logw, rng=RNG(1), exec=Executor())
    values_equal = all(x == y for x, y in zip(new_particles, expected))
    expected_logw = np.full((N,), -np.log(N))
    assert values_equal
    np.testing.assert_almost_equal(new_logw, expected_logw)


@pytest.mark.parametrize(
    "ncopies,expected",
    [(np.array([0, 2, 1, 0, 2, 1]), np.array([0, 2, 1, 0, 2, 1])),
     (np.array([0, 2, 2, 0, 2, 1]), np.array([0, 2, 2, 0, 2, 0])),
     (np.array([0, 2, 1, 0, 1, 1]), np.array([0, 2, 1, 0, 1, 2]))]
)
def test_check_stability(ncopies, expected):
    exec = Executor()
    first, last = np.array(split_across_cores(len(ncopies), exec.P, exec.rank))
    local_ncopies = ncopies[first:(last+1)]
    local_check_ncopies = check_stability(local_ncopies, exec)
    local_expected = expected[first:(last+1)]
    assert all(local_check_ncopies == local_expected)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "ncopies,expected",
    [(np.array([0, 2, 1, 0, 2, 1]), np.array([0, 2, 1, 0, 2, 1])),
     (np.array([0, 2, 2, 0, 2, 1]), np.array([0, 2, 2, 0, 2, 0])),
     (np.array([0, 2, 1, 0, 1, 1]), np.array([0, 2, 1, 0, 1, 2]))]
)
def test_check_stability_MPI(ncopies, expected):
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(len(ncopies), exec.P, exec.rank))
    local_ncopies = ncopies[first:(last+1)]
    local_check_ncopies = check_stability(local_ncopies, exec)
    local_expected = expected[first:(last+1)]
    assert all(local_check_ncopies == local_expected)

# TODO:
# get_number_of_copies
