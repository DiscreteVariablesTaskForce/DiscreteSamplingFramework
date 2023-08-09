import numpy as np
from scipy.special import logit, expit


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

# transforms a continuous unconstrained parameter to the constrained space for a given set of limit values
def u2c(unconstrained_param, min_val, max_val):
    return min_val+(max_val-min_val)*expit(unconstrained_param)


# transforms a continuous constrained parameter to the unconstrained space for a given set of limit values
def c2u(constrained_param, min_val, max_val):
    return logit((constrained_param - min_val) / (max_val - min_val))


# transforms arrays of continuous unconstrained parameters to constrained parameters for a given set of limit values
def u2c_array(unconstrained_params, min_vals, max_vals):
    constrained_params = []
    for param_array, min, max in zip(unconstrained_params, min_vals, max_vals):
        constrained_params.append(min+(max-min)*expit(param_array))
    return constrained_params


# transforms arrays of continuous constrained parameters to unconstrained parameters for a given set of limit values
def c2u_array(constrained_params, min_vals, max_vals):
    unconstrained_params = []
    for param_array, min, max in zip(constrained_params, min_vals, max_vals):
        unconstrained_params.append(logit((param_array-min)/(max-min)))
    return unconstrained_params

