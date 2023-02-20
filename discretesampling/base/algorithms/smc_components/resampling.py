from mpi4py import MPI
import numpy as np
from discretesampling.base.algorithms.smc_components.prefix_sum import inclusive_prefix_sum
from discretesampling.base.algorithms.smc_components.variable_size_redistribution.redistribution import redistribute


def get_number_of_copies(logw, rng):
    comm = MPI.COMM_WORLD
    N = len(logw) * comm.Get_size()

    cdf = inclusive_prefix_sum(np.exp(logw)*N)
    cdf_of_i_minus_one = cdf - np.reshape(np.exp(logw) * N, newshape=cdf.shape)

    u = np.array(rng.uniform(0.0, 1.0), dtype=logw.dtype)
    comm.Bcast(buf=[u, MPI._typedict[u.dtype.char]], root=0)

    ncopies = np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)
    return ncopies.astype(int)


def systematic_resampling(particles, logw, rng):
    loc_n = len(logw)
    N = loc_n * MPI.COMM_WORLD.Get_size()

    ncopies = get_number_of_copies(logw.astype('float32'), rng)
    particles = redistribute(particles, ncopies)
    logw = np.log(np.ones(loc_n) / N)

    return particles, logw
