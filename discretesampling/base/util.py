import numpy as np


def pad(x, exec):
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
    encoded_type = encoded_particles[0].dtype
    max_dim = exec.max(dims)
    paddings = [np.full((max_dim - dim), -1, encoded_type) for dim in dims]
    padded = np.vstack([np.hstack((particle, padding)) for (particle, padding) in zip(encoded_particles, paddings)])
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
    decoded_x = [particles[0].decode(encoded_particle, particles[0]) for encoded_particle in x]

    return decoded_x


def gather_all(particles, exec):
    loc_n = len(particles)
    N = loc_n * exec.P

    x = pad(particles, exec)

    all_particles = [particles[0] for i in range(N)]
    all_x_shape = [N, x.shape[1]]
    all_x = exec.gather(x, all_x_shape)
    all_particles = restore(all_x, all_particles)

    return all_particles
