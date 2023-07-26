import pytest
from discretesampling.domain.additive_structure.numbers import bell, stirling, binomial


@pytest.mark.parametrize(
    "n,k,expected",
    [(1, 1, 1),
     (2, 1, 2),
     (2, 2, 1),
     (3, 1, 3),
     (3, 2, 3),
     (3, 3, 1),
     (10, 5, 252),
     (1, 2, 0)]
)
def test_binomial(n, k, expected):
    binomial_n_k = binomial(n, k)
    assert binomial_n_k == expected


@pytest.mark.parametrize(
    "n,expected",
    [(0, 1),
     (1, 1),
     (2, 2),
     (3, 5),
     (4, 15),
     (5, 52),
     (6, 203),
     (7, 877),
     (8, 4140),
     (9, 21147)]
)
def test_bell(n, expected):
    bell_n = bell(n)
    assert bell_n == expected


@pytest.mark.parametrize(
    "n,k,expected",
    [(1, 1, 1),
     (2, 1, 1),
     (2, 2, 1),
     (3, 1, 1),
     (3, 2, 3),
     (3, 3, 1),
     (4, 1, 1),
     (4, 2, 7),
     (4, 3, 6),
     (10, 5, 42525)]
)
def test_stirling(n, k, expected):
    stirling_n_k = stirling(n, k)
    assert stirling_n_k == expected
