import numpy as np
from mpi4py import MPI

def max_dimension(x):
        comm = MPI.COMM_WORLD
        local_max = np.max(x)
        max_dim = np.zeros_like(1, dtype=local_max.dtype)
        comm.Allreduce(sendbuf=[local_max, MPI.INT], recvbuf=[max_dim, MPI.INT], op=MPI.MAX)
        return max_dim

def pad(x):
    """
    Description
    -----------
    This function computes the size of the biggest particle, and extend the other particles with NaNs until all
    particles have the same size

    :param x: particles organized as a list of objects
    :return x_new: particle organized as an encoded and padded numpy 2D array
    """
    
    encoded_particles = [x[0].encode(particle) for particle in x]
    dims = np.array([len(y) for y in encoded_particles])
    max_dim = max_dimension(dims)
    paddings = [np.full((max_dim - dim), -1) for dim in dims]
    padded = np.vstack([np.hstack((particle,padding)) for (particle, padding) in zip(encoded_particles, paddings)])
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
    decoded_x = [particles[0].decode(encoded_tree, particles[0]) for encoded_tree in x]

    return decoded_x


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

