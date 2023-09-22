from math import factorial


def binomial(n, k):
    """
    :return: Binomial coefficient
    """
    if 0 <= k <= n:
        return factorial(n) // factorial(k) // factorial(n - k)
    else:
        return 0


def stirling(n, k):
    """
    Stirling number general formula: S(n+1, k) = k*S(n, k) + S(n, k-1), where
    :param n: number of elements to be split
    :param k: number of sets
    :return: the stirling number of the k kind
    """
    if n == 0 or k == 0 or k > n:
        return 0
    if k == 1 or k == n:
        return 1
    return k*stirling(n-1, k) + stirling(n-1, k-1)


def bell(n):
    """
    Bell number general formula: B_{n+1} = sum_{k=0}^{n} [ binomial(n,k) * B_n ]  where  # noqa
    :param n: number of elements
    :return: the nth Bell number (number of partitions of n elements)
    alternatively B_{n} = sum_{k=0}^{n} [stirling(n,k)]
    """
    if n == 1 or n == 0:
        return 1
    else:
        return sum([stirling(n, k) for k in range(0, n+1)])
