import pytest
import numpy as np
from discretesampling.base.executor import Executor


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
