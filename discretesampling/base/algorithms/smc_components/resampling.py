import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base.executor.executor import Executor


def check_stability(ncopies, exec=Executor()):

    loc_n = len(ncopies)
    N = loc_n * exec.P
    rank = exec.rank

    sum_of_ncopies = exec.sum(ncopies)

    if sum_of_ncopies != N:
        # Find the index of the last particle to be copied
        idx = np.where(ncopies > 0)
        idx = idx[0][-1]+rank*loc_n if len(idx[0]) > 0 else np.array([-1])
        max_idx = exec.max(idx)
        # Find the core which has that particle, and increase/decrease its ncopies[i] till sum_of_ncopies == N
        if rank*loc_n <= max_idx <= (rank + 1)*loc_n - 1:
            ncopies[max_idx - rank*loc_n] -= sum_of_ncopies - N

    return ncopies


def get_number_of_copies(logw, rng=RNG(), exec=Executor()):
    N = len(logw) * exec.P

    cdf = calculate_cdf(logw, exec)
    cdf_of_i_minus_one = cdf - np.reshape(np.exp(logw) * N, newshape=cdf.shape)

    u = np.array(rng.uniform(0.0, 1.0), dtype=logw.dtype)
    exec.bcast(u)
    ncopies = (np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)).astype(int)
    ncopies = check_stability(ncopies, exec)

    return ncopies  # .astype(int)


def calculate_cdf(logw, exec=Executor()):
    N = len(logw) * exec.P
    return np.exp(exec.logcumsumexp(logw)) * N


def systematic_resampling(particles, logw, rng, exec=Executor()):
    loc_n = len(logw)
    N = loc_n * exec.P

    ncopies = get_number_of_copies(logw.astype('float32'), rng, exec)
    particles = exec.redistribute(particles, ncopies)
    logw = np.log(np.ones(loc_n) / N)

    return particles, logw
