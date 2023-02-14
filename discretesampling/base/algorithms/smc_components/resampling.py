from mpi4py import MPI
import numpy as np
import math
from discretesampling.base.algorithms.smc_components.prefix_sum import inclusive_prefix_sum
from discretesampling.base.algorithms.smc_components.redistribution import centralised_redistribution
from discretesampling.base.algorithms.smc_components.logsumexp import log_sum_exp
from scipy.special import logsumexp


def get_number_of_copies(logw, rng):
    comm = MPI.COMM_WORLD
    N = len(logw) * comm.Get_size()

    cdf = inclusive_prefix_sum(np.exp(logw)*N)

    u = np.array(rng.randomInt(0, N-1), dtype='d')  # u = np.array(rng.randomInt(0, 32766), dtype='d')  # np.array(np.random.uniform(0.0, 32767), dtype='d')
    comm.Bcast(buf=[u, MPI.DOUBLE], root=0)
    u = u/N

    cdf_of_i_minus_one = cdf - np.reshape(np.exp(logw)*N, newshape=cdf.shape)

    ncopies = np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)
    return ncopies.astype(int)


def resample(particles, logWeights, rng):
    N = len(particles)

    new_indexes = rng.randomChoices(population=range(N), weights=np.exp(logWeights), k=N)
    new_particles = [particles[i] for i in new_indexes]
    new_logWeights = np.full(N, -math.log(N))

    return new_particles, new_logWeights


def systematic_resampling(particles, logw, rng):
    loc_n = len(logw)
    comm = MPI.COMM_WORLD
    N = loc_n * MPI.COMM_WORLD.Get_size()

    ncopies = get_number_of_copies(logw, rng)
    particles = centralised_redistribution(particles, ncopies)
    logw = np.log(np.ones(loc_n) / N)

    return particles, logw
