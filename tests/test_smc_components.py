import pytest
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base.executor.executor import Executor
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling, check_stability


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


@pytest.mark.parametrize(
    "logw,expected_ess",
    [(np.array([-np.log(8)] * 8), 8.0),  # 8 even weights, 0.5,0.5
     (np.array([-np.log(1.0)] + [-np.inf] * 7), 1.0)]  # two weights, 1,0
)
def test_ess(logw, expected_ess):
    calc_ess = ess(logw, exec=Executor())
    np.testing.assert_allclose(calc_ess, expected_ess)  # use almost_equal for numerical inaccuracy


@pytest.mark.parametrize(
    "array,expected",
    [(np.array([0.0]*8), np.array([-np.log(8)]*8)),  # equal, non-normalised weights
     (np.array([-np.log(16)]*2 + [np.log(9/16)] + [-np.log(16)]*5),
      np.array([-np.log(16)]*2 + [np.log(9/16)] + [-np.log(16)]*5))  # already normalised
     ]
)
def test_log_normalise(array, expected):
    normalised_array = normalise(array, exec=Executor())
    np.testing.assert_allclose(normalised_array, expected)  # use allclose for numerical inaccuracy


@pytest.mark.parametrize(
    "particles,logw,expected",

    [([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]], np.array([-np.log(8)]*8),
      [ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]]),
     ([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]], np.array([np.log(1)] + [-np.inf]*7),
      [ExampleParticleClass(x) for x in [1, 1, 1, 1, 1, 1, 1, 1]]),
     ([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]], np.array([np.log(0.5), np.log(0.5)] + [-np.inf]*6),
      [ExampleParticleClass(x) for x in [1, 1, 1, 1, 2, 2, 2, 2]]),
     ([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]],
      np.array([np.log(0.25), -np.inf, np.log(0.25), -np.inf] * 2),
      [ExampleParticleClass(x) for x in [1, 1, 3, 3, 5, 5, 7, 7]])]
)
def test_systematic_resampling(particles, logw, expected):
    N = len(particles)
    new_particles, new_logw = systematic_resampling(particles, logw, rng=RNG(1), exec=Executor())
    values_equal = all(x == y for x, y in zip(new_particles, expected))
    expected_logw = np.full((N,), -np.log(N))
    assert values_equal
    np.testing.assert_almost_equal(new_logw, expected_logw)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "particles,logw,expected",

    [([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]], np.array([-np.log(8)]*8),
      [ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]]),
     ([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]], np.array([np.log(1)] + [-np.inf]*7),
      [ExampleParticleClass(x) for x in [1, 1, 1, 1, 1, 1, 1, 1]]),
     ([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]], np.array([np.log(0.5), np.log(0.5)] + [-np.inf]*6),
      [ExampleParticleClass(x) for x in [1, 1, 1, 1, 2, 2, 2, 2]]),
     ([ExampleParticleClass(x) for x in [1, 2, 3, 4, 5, 6, 7, 8]],
      np.array([np.log(0.25), -np.inf, np.log(0.25), -np.inf] * 2),
      [ExampleParticleClass(x) for x in [1, 1, 3, 3, 5, 5, 7, 7]])]
)
def test_systematic_resampling_MPI(particles, logw, expected):
    N = len(particles)
    exec = Executor_MPI()
    first, last = np.array(split_across_cores(N, exec.P, exec.rank))
    local_particles = particles[first:(last+1)]
    local_logw = logw[first:(last+1)]
    new_local_particles, new_local_logw = systematic_resampling(local_particles, local_logw, rng=RNG(1), exec=exec)
    expected_local_particles = expected[first:(last+1)]
    values_equal = all(x == y for x, y in zip(new_local_particles, expected_local_particles))
    expected_logw = np.full((N,), -np.log(N))
    expected_local_logw = expected_logw[first:(last+1)]
    assert values_equal
    np.testing.assert_almost_equal(new_local_logw, expected_local_logw)


@pytest.mark.parametrize(
    "ncopies,expected",
    [(np.array([0, 2, 1, 0, 2, 1, 1, 1]), np.array([0, 2, 1, 0, 2, 1, 1, 1])),
     (np.array([0, 2, 2, 0, 2, 1, 1, 1]), np.array([0, 2, 2, 0, 2, 1, 1, 0])),
     (np.array([0, 2, 1, 0, 1, 1, 1, 1]), np.array([0, 2, 1, 0, 1, 1, 1, 2]))]
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
    [(np.array([0, 2, 1, 0, 2, 1, 1, 1]), np.array([0, 2, 1, 0, 2, 1, 1, 1])),
     (np.array([0, 2, 2, 0, 2, 1, 1, 1]), np.array([0, 2, 2, 0, 2, 1, 1, 0])),
     (np.array([0, 2, 1, 0, 1, 1, 1, 1]), np.array([0, 2, 1, 0, 1, 1, 1, 2]))]
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
