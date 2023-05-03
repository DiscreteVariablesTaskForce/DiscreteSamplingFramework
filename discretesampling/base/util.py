import numpy as np
from mpi4py import MPI
from discretesampling.domain.decision_tree.tree import Tree


def pad(x):
    """
    Description
    -----------
    This function computes the size of the biggest particle, and extend the other particles with NaNs until all
    particles have the same size

    :param x: particles organized as a list of lists
    :return x_new: particle organized as a numpy 2D array
    """
    def max_dimension(x):
        comm = MPI.COMM_WORLD
        local_max = np.max(x)
        max_dim = np.zeros_like(1, dtype=local_max.dtype)
        comm.Allreduce(sendbuf=[local_max, MPI.INT], recvbuf=[max_dim, MPI.INT], op=MPI.MAX)
        return max_dim
    
    encoded_particles = x[0].encode(x)
    dims = np.array([len(y) for y in encoded_particles])
    max_dim = max_dimension(dims)
    padded = np.hstack((encoded_particles, np.atleast_2d(max_dim).transpose()))
    return padded
    
def restore(x, particles):
    """
    Description
    -----------
    This function unpacks padded particles and decodes them

    :param x: encoded, padded particles
    :return decoded_x
    """ 
    # remove padding
    restored_x = x
    decoded_x = particles[0].decode(x, particles)
    decoded_x


def gather_all(particles):
    comm = MPI.COMM_WORLD
    loc_n = len(particles)
    N = loc_n * comm.Get_size()

    x = pad(particles)
    all_x = np.zeros([N, x.shape[1]], dtype='d')
    all_particles = [particles[0] for i in range(N)]

    comm.Allgather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE])

    all_particles = restore(all_x, all_particles)

    return all_particles

