import pytest
import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.executor.executor import Executor


class ExampleParticleClass(DiscreteVariable):
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x

    def getProposalType(self):
        return super().getProposalType()

    def getTargetType(self):
        return super().getTargetType()


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5]), 5),
     (np.array([1, 1, 1, 1, 1]), 1),
     (np.array([0.0, 0.0, 1.23]), 1.23)]
)
def test_executor_max(x, expected):
    max_x = Executor().max(x)
    assert expected == max_x


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5]), 15),
     (np.array([1, 1, 1, 1, 1]), 5),
     (np.array([0.0, 0.0, 1.23]), 1.23)]
)
def test_executor_sum(x, expected):
    sum_x = Executor().sum(x)
    assert expected == sum_x


@pytest.mark.parametrize(
    "x",
    [(np.array([1, 2, 3, 4, 5])),
     (np.array([1, 1, 1, 1, 1])),
     (np.array([0.0, 0.0, 1.23]))]
)
def test_executor_gather(x):
    all_x_shape = x.shape
    all_x = Executor().gather(x, all_x_shape)
    assert all(x == all_x)


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([np.log(1.0)]), 0.0),  # single weight
     (np.array([np.log(0.5), np.log(0.5)]), 0.0),  # two even weights, 0.5,0.5
     (np.array([np.log(1.0), -np.inf]), 0.0),  # two weights, 1,0
     (np.array([np.log(1e5), np.log(1e-5), -np.inf]), 11.512925465070229)]
)
def test_logsumexp(x, expected):
    calc = Executor().logsumexp(x)
    np.testing.assert_almost_equal(calc, expected)  # use almost_equal for numerical inaccuracy


@pytest.mark.parametrize(
    "x,expected",
    [(np.array([1, 2, 3, 4, 5]), np.array([1, 3, 6, 10, 15])),
     (np.array([-1, -2, -3, -4, -5]), np.array([-1, -3, -6, -10, -15]))]
)
def test_cumsum(x, expected):
    calc = Executor().cumsum(x)
    assert all(calc == expected)


@pytest.mark.parametrize(
    "particles,ncopies,expected",
    [([ExampleParticleClass(x) for x in ["a", "b", "c", "d", "e"]], np.array(
        [0, 2, 1, 0, 2]), [ExampleParticleClass(x) for x in ["b", "b", "c", "e", "e"]])]
)
def test_redistribute(particles, ncopies, expected):
    new_particles = Executor().redistribute(particles, ncopies)
    assert new_particles == expected
