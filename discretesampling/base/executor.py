import itertools
from mpi4py import MPI
import numpy as np
from discretesampling.base.algorithms.smc_components.distributed_fixed_size_redistribution.prefix_sum import (
    inclusive_prefix_sum
)
from discretesampling.base.algorithms.smc_components.variable_size_redistribution import (
    variable_size_redistribution
)


class Executor(object):
    def __init__(self):
        self.P = 1
        self.rank = 1

    def max(self, x):
        return np.max(x)

    def sum(self, x):
        return np.sum(x)

    def gather(self, x, all_x_shape):
        return x

    def bcast(self, x):
        pass

    def cumsum(self, x):
        return np.cumsum(x)

    def redistribute(self, particles, ncopies):
        particles = list(itertools.chain.from_iterable(
            [[particles[i]]*ncopies[i] for i in range(len(particles))]
        ))
        return particles


class Executor_MPI(Executor):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()  # number of MPI nodes/ranks
        self.rank = self.comm.Get_rank()

    def max(self, x):
        local_max = np.max(x)
        x_dtype = MPI._typedict[x.dtype.char]
        max_dim = np.zeros_like(1, dtype=x_dtype)
        self.comm.Allreduce(sendbuf=[local_max, MPI.INT], recvbuf=[max_dim, MPI.INT], op=MPI.MAX)
        return max_dim

    def sum(self, x):
        x_dtype = MPI._typedict[x.dtype.char]
        sum_of_x = np.array(1, dtype=x.dtype)
        self.comm.Allreduce(sendbuf=[np.sum(x), x_dtype], recvbuf=[sum_of_x, x_dtype], op=MPI.SUM)
        return sum_of_x

    def gather(self, x, all_x_shape):
        x_dtype = MPI._typedict[x.dtype.char]
        all_x = np.zeros(all_x_shape, dtype=x.dtype)
        self.comm.Allgather(sendbuf=[x, x_dtype], recvbuf=[all_x, x_dtype])
        return all_x

    def bcast(self, x):
        self.comm.Bcast(buf=[x, MPI._typedict[x.dtype.char]], root=0)

    def cumsum(self, x):
        return inclusive_prefix_sum(x)

    def redistribute(self, particles, ncopies):
        return variable_size_redistribution(particles, ncopies)
