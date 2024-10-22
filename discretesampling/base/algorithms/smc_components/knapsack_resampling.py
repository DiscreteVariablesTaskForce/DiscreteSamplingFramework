import numpy as np
from discretesampling.base.algorithms.smc_components.resampling import check_stability
from discretesampling.base.executor import Executor


def knapsack(W, wt, val, n):
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    res = K[n][W]

    zero_one_copies = np.zeros(n).astype('i')

    w = W
    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == K[i - 1][w]:
            continue
        else:
            zero_one_copies[i-1] = 1

            # Since this weight is included its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]

    return zero_one_copies, K[n][W]


def knapsack_resampling(x, w, mvrs_rng, exec=Executor()):
    x_new = []
    E = len(w)
    T = mvrs_rng.uniform(low=0.01, high=1.0)
    max_w = np.max(w)
    if max_w > T:
        T = max_w
    C = int(E*T)
    N = len(w)

    w_knap = np.ceil(E * w).astype(int)
    #w_knap[-1] = 53
    value = (w_knap ** 2).astype(int)

    zero_one_copies, res = knapsack(W=C, wt=w_knap, val=value, n=N)

    wsum = np.sum(zero_one_copies * w)

    ncopies = (N * (zero_one_copies * w)/wsum + 0.5).astype(int)
    ncopies = check_stability(ncopies, exec)
    #x_new = np.repeat(x, ncopies)
    i = 0
    for j in range(len(ncopies)):
        for k in range(ncopies[j]):
            i += 1
            x_new.append(x[j])
    log_new_w = np.repeat(np.log(1/N), N).astype('float32')

    return x_new, log_new_w, ncopies

# """
# profit = [60, 100, 120]
# weight = [10, 20, 30]
# W = 50
# n = len(profit)
# zero_one_copies, res = knapsack(W, weight, profit, n)
#
# print(zero_one_copies)
# print(res)
# """
#
# w = np.array([1.0, 2, 3, 8, 16])/30
# x = np.array([1.2, 2.3, 3.4, 4.5, 6.0])
#
# new_x, new_w, ncopies = knapsack_resampling(x, w)
