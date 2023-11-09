from mpi4py import MPI
import numpy as np
from scipy.special import logsumexp
from discretesampling.base.executor import Executor
from smccomponents.resample.mpi.prefix_sum import inclusive_prefix_sum
from discretesampling.base.executor.MPI.variable_size_redistribution import (
    variable_size_redistribution
)


def LSE(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype='d')
    y = np.frombuffer(ymem, dtype='d')
    y[:] = logsumexp(np.hstack((x, y)))


class Executor_MPI(Executor):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()  # number of MPI nodes/ranks
        self.rank = self.comm.Get_rank()

    def max(self, x):
        local_max = np.max(x)
        x_dtype = MPI._typedict[x.dtype.char]
        max_dim = np.zeros_like(1, dtype=x.dtype)
        self.comm.Allreduce(sendbuf=[local_max, x_dtype], recvbuf=[max_dim, x_dtype], op=MPI.MAX)
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

    def logsumexp(self, x):
        op = MPI.Op.Create(LSE, commute=True)
        log_sum = np.zeros_like(1, x.dtype)
        MPI_dtype = MPI._typedict[x.dtype.char]
        leaf_node = np.array([-np.inf]).astype(x.dtype) if len(x) == 0 else logsumexp(x)

        MPI.COMM_WORLD.Allreduce(sendbuf=[leaf_node, MPI_dtype], recvbuf=[log_sum, MPI_dtype], op=op)

        op.Free()

        return log_sum

    def cumsum(self, x):
        return inclusive_prefix_sum(x)

    def redistribute(self, particles, ncopies):
        return variable_size_redistribution(particles, ncopies, self)
