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
    my_tree_dims = np.array([len(tree) for tree in trees])
    my_leaf_dims = np.array([len(leaf) for leaf in leaves])
    max_tree = max_dimension(x=my_tree_dims)
    max_leaf = max_dimension(x=my_leaf_dims)

    x_new = np.array([np.hstack((tree, np.repeat(-1.0, max_tree - len(tree)), last_action,
                                 leaf, np.repeat(-1.0, max_leaf - len(leaf)), max_tree))
                      for tree, last_action, leaf in zip(trees, last_actions, leaves)])
    return np.hstack((x_new, np.atleast_2d(my_tree_dims).transpose(), np.atleast_2d(my_leaf_dims).transpose()))


def restore(x, particles):
    my_leaf_dim = x[:, -1].astype(int)
    my_tree_dim = x[:, -2].astype(int)
    max_tree = int(x[0, -3])
    for i in range(len(particles)):
        particles[i].set_tree(x[i, 0:my_tree_dim[i]])
        particles[i].set_lastAction(x[i, max_tree])
        particles[i].set_leafs(x[i, max_tree+1:max_tree+1+my_leaf_dim[i]])


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
        print(string, all_ncopies.tolist())


def print_particles_to_file(ncopies, x, filename="particles.py"):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    N = loc_n * comm.Get_size()
    all_ncopies = np.zeros(N, dtype=ncopies.dtype)
    all_x = np.zeros([N, x.shape[1]], dtype='d')

    comm.Gather(sendbuf=[ncopies, MPI.INT], recvbuf=[all_ncopies, MPI.INT], root=0)
    comm.Gather(sendbuf=[x, MPI.DOUBLE], recvbuf=[all_x, MPI.DOUBLE], root=0)

    if comm.Get_rank() == 0:
        src = open(filename, 'w')
        src.write('import numpy as np\n')
        src.write('nan = np.nan\n')
        src.write('all_ncopies = np.array('+str(all_ncopies.tolist())+')\n\n')
        src.write('all_x = np.array(['+str(all_x[0, :].tolist())+',\n')
        for i in range(1, N-1):
            src.write('                  ' +str(all_x[i, :].tolist())+',\n')
        src.write('                  '+str(all_x[-1, :].tolist())+'])\n')
        src.close()

        #src.write(' '.join(map(str, all_ncopies.tolist())))
        src.close()
