from math import factorial


def binomial(n: int, k: int) -> int:
    """Calculate binomial coefficient

    Parameters
    ----------
    n : int
        _description_
    k : int
        _description_

    Returns
    -------
    int
        Binomial coefficient
    """
    if 0 <= k <= n:
        return factorial(n) // factorial(k) // factorial(n - k)
    else:
        return 0


def stirling(n, k):
    """Calculate Stirling number.

    Parameters
    ----------
    n : int
        Number of elements
    k : int
        Number of sets

    Returns
    -------
    int
        S(n,k), number of possible partitions of n elements into k nonempty subsets


    Notes
    -----
    Stirling numbers of the second kind are generated recursively following the formula:
    .. math:: S(n+1, k) = k*S(n, k) + S(n, k-1)
    where:
    .. math:: S(0,k) = S(n,0) = 0
    """
    if n == 0 or k == 0 or k > n:
        return 0
    if k == 1 or k == n:
        return 1
    return k*stirling(n-1, k) + stirling(n-1, k-1)


def bell(n: int) -> int:
    """Calculate the nth bell number

    Parameters
    ----------
    n : int
        Number of elements

    Returns
    -------
    int
        The nth Bell number (number of partitions of n elements)

    Notes
    -----
    .. ::math B_{n+1} = sum_{k=0}^{n} [ binomial(n,k) * B_n ]
    .. ::math B_{n} = sum_{k=0}^{n} [stirling(n,k)]

    """
    if n == 1 or n == 0:
        return 1
    else:
        return sum([stirling(n, k) for k in range(0, n+1)])
