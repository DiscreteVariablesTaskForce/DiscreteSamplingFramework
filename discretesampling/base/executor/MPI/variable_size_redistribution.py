from discretesampling.base.util import pad, restore
#from smccomponents.resample.mpi.redistribution import (
#    fixed_size_redistribution
#)

def sequential_redistribution(x, ncopies):
    return np.repeat(x, ncopies, axis=0)
def fixed_size_redistribution(x, ncopies):

    if MPI.COMM_WORLD.Get_size() > 1:
        x, ncopies = rot_nearly_sort(x, ncopies)
        x, ncopies = rot_split(x, ncopies)

    x = sequential_redistribution(x, ncopies)

    return x

def inclusive_prefix_sum(array):
    comm = MPI.COMM_WORLD

    csum = np.cumsum(array).astype(array.dtype)
    offset = np.zeros(1, dtype=array.dtype)
    MPI_dtype = MPI._typedict[array.dtype.char]
    comm.Exscan(sendbuf=[csum[-1], MPI_dtype], recvbuf=[offset, MPI_dtype], op=MPI.SUM)

    return csum + offset


def variable_size_redistribution(particles, ncopies, exec):
    x = pad(particles, exec)

    x = fixed_size_redistribution(x, ncopies)

    particles = restore(x, particles)

    return particles
