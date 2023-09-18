import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.executor import Executor


def pad(x: list[DiscreteVariable], exec: Executor = Executor()) -> np.ndarray:
    """Encode a list of particles to a :class numpy.ndarray: and pad to size
    of the largest particle.

    Parameters
    ----------
    x : list[DiscreteVariable]
        List of particles to encode.
    exec : Executor, optional
        Executor engine, by default Executor()

    Returns
    -------
    np.ndarray
        Numpy array containing encoded and padded particles.
    """
    encoded_particles = [x[0].encode(particle) for particle in x]
    dims = np.array([len(y) for y in encoded_particles])
    encoded_type = encoded_particles[0].dtype
    max_dim = exec.max(dims)
    paddings = [np.full((max_dim - dim), -1, encoded_type) for dim in dims]
    padded = np.vstack([np.hstack((particle, padding)) for (particle, padding) in zip(encoded_particles, paddings)])
    return padded


def restore(x: np.ndarray, particles: list[DiscreteVariable]) -> list[DiscreteVariable]:
    """Unpack and decode an array of encoded particles.

    Parameters
    ----------
    x : np.ndarray
        Numpy array of encoded particles.
    particles : list[DiscreteVariable]
        List of particles of type which `x` should be decoded to.

    Returns
    -------
    list[DiscreteVariable]
        List of decoded particles.
    """
    decoded_x = [particles[0].decode(encoded_particle, particles[0]) for encoded_particle in x]

    return decoded_x


def gather_all(particles: list[DiscreteVariable], exec: Executor = Executor()):
    """Gather particles from all nodes of an executor engine.

    Parameters
    ----------
    particles : list[DiscreteVariable]
        _description_
    exec : Executor, optional
        _description_, by default Executor()

    Returns
    -------
    _type_
        _description_
    """
    loc_n = len(particles)
    N = loc_n * exec.P

    x = pad(particles, exec)

    all_particles = [particles[0] for i in range(N)]
    all_x_shape = [N, x.shape[1]]
    all_x = exec.gather(x, all_x_shape)
    all_particles = restore(all_x, all_particles)

    return all_particles
