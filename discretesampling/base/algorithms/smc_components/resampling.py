from mpi4py import MPI
import numpy as np
import math
from discretesampling.base.algorithms.smc_components.prefix_sum import inclusive_prefix_sum
from discretesampling.base.algorithms.smc_components.util import pad, restore


def get_number_of_copies(logw, rng):
    comm = MPI.COMM_WORLD
    N = len(logw) * comm.Get_size()

    cdf = inclusive_prefix_sum(np.exp(logw)*N)

    u = np.array(rng.randomInt(0, 32766), dtype='d')  # np.array(np.random.uniform(0.0, 32767), dtype='d')
    comm.Bcast(buf=[u, MPI.DOUBLE], root=0)
    u = u/N

    cdf_of_i_minus_one = cdf - np.reshape(np.exp(logw)*N, newshape=cdf.shape)

    ncopies = np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)
    return ncopies.astype(int)


def sequential_redistribution(x, ncopies):
    N = len(ncopies)
    x_new = np.zeros_like(x)
    i = 0
    for j in range(N):
        for k in range(ncopies[j]):
            x_new[i] = x[j]
            i += 1
    return x_new


def centralised_redistribution(x, ncopies):
    comm = MPI.COMM_WORLD
    N = len(ncopies) * comm.Get_size()
    rank = comm.Get_rank()

    all_ncopies = np.zeros(N, dtype='i')
    all_x = np.zeros([N, x.shape[1]], dtype='d')

    comm.Gather(sendbuf=[ncopies, MPI.INT], recvbuf=[all_ncopies, MPI.INT], root=0)
    comm.Gather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE], root=0)

    if rank == 0:
        all_x = sequential_redistribution(all_x, all_ncopies)

    comm.Scatter(sendbuf=[all_x, MPI.DOUBLE], recvbuf=[x, MPI.DOUBLE], root=0)

    return x


def resample(particles, logWeights, rng):
    N = len(particles)

    new_indexes = rng.randomChoices(population=range(N), weights=np.exp(logWeights), k=N)
    new_particles = [particles[i] for i in new_indexes]
    new_logWeights = np.full(N, -math.log(N))

    return new_particles, new_logWeights


def minimum_variance_resampling(particles, logw, rng):
    loc_n = len(logw)
    N = loc_n * MPI.COMM_WORLD.Get_size()
    x = pad(particles)
    #x = convert(x)
    ncopies = get_number_of_copies(logw, rng)
    x_new = centralised_redistribution(x, ncopies)
    logw = np.log(np.ones(loc_n) / N)
    #x_new, logw = resample(x, logw, rng)
    #x_new = convert(x_new)
    restore(x_new, particles)
    return particles, logw
