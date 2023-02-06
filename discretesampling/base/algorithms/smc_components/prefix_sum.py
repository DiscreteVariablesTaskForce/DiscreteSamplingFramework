from mpi4py import MPI
import numpy as np


def inclusive_prefix_sum(array):
    comm = MPI.COMM_WORLD

    csum = np.cumsum(array)
    offset = np.zeros(1, dtype=array.dtype)
    if array.dtype == 'd':
        MPI_dtype = MPI.DOUBLE
    elif array.dtype == 'i':
        MPI_dtype = MPI.INT
    else:
        MPI_dtype = MPI.DOUBLE
    comm.Exscan(sendbuf=[csum[-1], MPI_dtype], recvbuf=[offset, MPI_dtype], op=MPI.SUM)

    return csum + offset
