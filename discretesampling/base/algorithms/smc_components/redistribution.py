import numpy as np
from mpi4py import MPI
from discretesampling.base.algorithms.smc_components.util import pad, restore, print_all


def sequential_redistribution(x, ncopies):
    return np.repeat(x, ncopies, axis=0)


def centralised_redistribution(particles, ncopies):
    comm = MPI.COMM_WORLD
    N = len(ncopies) * comm.Get_size()
    rank = comm.Get_rank()

    x = pad(particles)

    all_ncopies = np.zeros(N, dtype=ncopies.dtype)
    all_x = np.zeros([N, x.shape[1]], dtype='d')

    comm.Gather(sendbuf=[ncopies, MPI.INT], recvbuf=[all_ncopies, MPI.INT], root=0)
    comm.Gather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE], root=0)

    if rank == 0:
        all_x = sequential_redistribution(all_x, all_ncopies)

    comm.Scatter(sendbuf=[all_x, MPI.DOUBLE], recvbuf=[x, MPI.DOUBLE], root=0)

    restore(x, particles)

    return particles

