import pytest
import numpy as np
from discretesampling.base.random import RNG


@pytest.mark.parametrize(
    "seed,expected",
    [(0, 0.6369616873214543),
     (1, 0.5118216247002567),
     (2, 0.2616121342493164)]
)
def test_rng_random(seed, expected):
    x = RNG(seed).random()
    assert x == expected


@pytest.mark.parametrize(
    "seed,low,high,expected",
    [(0, 1, 10, 9),
     (1, 1, 1, 1),
     (2, 10, 100, 86)]
)
def test_rng_randomInt(seed, low, high, expected):
    x = RNG(seed).randomInt(low, high)
    assert x == expected


@pytest.mark.parametrize(
    "seed,low,high,expected",
    [(0, 1.0, 10.1, 6.796351354625234),
     (1, 1.0, 1.0, 1.0),
     (2, 0.1, 99.9, 26.20889099808178)]
)
def test_rng_uniform(seed, low, high, expected):
    x = RNG(seed).uniform(low, high)
    assert x == expected


@pytest.mark.parametrize(
    "seed,choices,expected",
    [(0, [1, 2, 3, 4, 5], 5),
     (1, ["a", "b", "c", "d", "e"], "c"),
     (2, [i for i in range(1, 100, 4)], 81)]
)
def test_rng_randomChoice(seed, choices, expected):
    x = RNG(seed).randomChoice(choices)
    assert x == expected


@pytest.mark.parametrize(
    "seed,choices,weights,k,expected",
    [(0, [1, 2, 3, 4, 5], [0.2 for i in range(5)], 2, np.array([4, 2])),
     (1, ["a", "b", "c", "d", "e"], [1.0, 0.0, 0.0, 0.0, 0.0], 3, np.array(['a']*3)),
     (2, [i for i in range(1, 100, 4)], [1/25 for i in range(1, 100, 4)], 3, np.array([25, 29, 81]))]
)
def test_rng_randomChoices(seed, choices, weights, k, expected):
    x = RNG(seed).randomChoices(choices, weights=weights, k=k)
    assert all(x == expected)
