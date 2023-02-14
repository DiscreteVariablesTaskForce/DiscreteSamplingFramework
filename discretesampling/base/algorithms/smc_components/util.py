import numpy as np
from mpi4py import MPI


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

    def encode_move(last):
        if last == "grow":
            return 0
        elif last == "prune":
            return 1
        elif last == "swap":
            return 2
        elif last == "change":
            return 3
        else:
            return -1

    trees = [np.array(particle.tree).flatten() for particle in x]
    leaves = [np.array(particle.leafs).flatten() for particle in x]
    last_actions = [encode_move(particle.lastAction) for particle in x]
    max_tree = max_dimension(x=np.array([len(tree) for tree in trees]))
    max_leaf = max_dimension(x=np.array([len(leaf) for leaf in leaves]))

    x_new = [np.hstack((tree, np.repeat(np.nan, max_tree - len(tree)), last_action,
                       leaf, np.repeat(np.nan, max_leaf - len(leaf)), max_tree, max_leaf))
             for tree, last_action, leaf in zip(trees, last_actions, leaves)]
    return np.array(x_new)


def restore(x, particles):
    max_tree = int(x[0, -2])
    for i in range(len(particles)):
        particles[i].set_tree(x[i, 0:max_tree])
        particles[i].set_lastAction(x[i, max_tree])
        particles[i].set_leafs(x[i, max_tree+1:-2])


def convert(x):
    max_tree = int(x[0, -2])
    return [[x[i, 0:max_tree], x[i, max_tree], x[i, max_tree+1:-2]] for i in range(x.shape[0])]


def gather_all(particles):
    comm = MPI.COMM_WORLD
    loc_n = len(particles)
    N = loc_n * comm.Get_size()

    x = pad(particles)
    all_x = np.zeros([N, x.shape[1]], dtype='d')
    all_particles = [particles[0] for i in range(N)]
    comm.Allgather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE])

    restore(all_x, all_particles)

    return all_particles


def print_all(x, ncopies):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    N = loc_n * comm.Get_size()
    all_x = np.zeros([N, x.shape[1]], dtype='d')
    all_ncopies = np.zeros(N, dtype=ncopies.dtype)

    comm.Gather(sendbuf=[ncopies, MPI.INT], recvbuf=[all_ncopies, MPI.INT], root=0)
    comm.Gather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE], root=0)

    if comm.Get_rank() == 0:
        print(all_x.transpose())
        print(all_ncopies)


def print_ncopies(ncopies, string=""):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    N = loc_n * comm.Get_size()
    all_ncopies = np.zeros(N, dtype=ncopies.dtype)

    comm.Gather(sendbuf=[ncopies, MPI.INT], recvbuf=[all_ncopies, MPI.INT], root=0)

    if comm.Get_rank() == 0:
        print(string, all_ncopies)
