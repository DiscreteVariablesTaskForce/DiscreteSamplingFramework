from typing import Union
import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.random import RNG
from discretesampling.base.executor.executor import Executor


def check_stability(ncopies: Union[list[int], np.ndarray], exec: Executor = Executor()) -> np.ndarray:
    """Check stability of ncopies

    Check that ncopies does not induce a change in the number of particles.

    Parameters
    ----------
    ncopies : Union[list[int], np.ndarray]
        Number of copies of particles for resampling
    exec : Executor, optional
        Execution engine, by default Executor()

    Returns
    -------
    np.ndarray
        Stabilised ncopies

    Notes
    -----
        Numerical inaccuracy in resampling can induce a change in the number of copies of particles.
        This function enforces that `len(ncopies) == sum(ncopies)`.
    """
    loc_n = len(ncopies)
    N = loc_n * exec.P
    rank = exec.rank

    sum_of_ncopies = exec.sum(ncopies)

    if sum_of_ncopies != N:
        # Find the index of the last particle to be copied
        idx = np.where(ncopies > 0)
        idx = idx[0][-1]+rank*loc_n if len(idx[0]) > 0 else np.array([-1])
        max_idx = exec.max(idx)
        # Find the core which has that particle, and increase/decrease its ncopies[i] till sum_of_ncopies == N
        if rank*loc_n <= max_idx <= (rank + 1)*loc_n - 1:
            ncopies[max_idx - rank*loc_n] -= sum_of_ncopies - N

    return ncopies


def get_number_of_copies(logw: Union[list[float], np.ndarray], rng: RNG = RNG(), exec: Executor = Executor()) -> np.ndarray:
    """Given logged weights calculate number of copies for resampling

    Parameters
    ----------
    logw : Union[list[float], np.ndarray]
        Logged importance weights
    rng : RNG, optional
        RNG for random number generation, by default RNG()
    exec : Executor, optional
        Execution engine, by default Executor()

    Returns
    -------
    np.ndarray
        Number of copies
    """
    N = len(logw) * exec.P

    cdf = exec.cumsum(np.exp(logw)*N)
    cdf_of_i_minus_one = cdf - np.reshape(np.exp(logw) * N, newshape=cdf.shape)

    u = np.array(rng.uniform(0.0, 1.0), dtype=logw.dtype)
    exec.bcast(u)
    ncopies = (np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)).astype(int)
    ncopies = check_stability(ncopies, exec)

    return ncopies  # .astype(int)


def systematic_resampling(
    particles: list[DiscreteVariable],
    logw: Union[list[float], np.ndarray], rng: RNG = RNG(),
    exec: Executor = Executor()
) -> tuple[list[DiscreteVariable], np.ndarray]:
    """Perform systematic resampling of particles given logged importance weights

    Parameters
    ----------
    particles : list[DiscreteVariable]
        List of particles to be resampled
    logw : Union[list[float], np.ndarray]
        Logged importance weights
    rng : RNG, optional
        RNG for random number generation, by default RNG()
    exec : Executor, optional
        Execution engine, by default Executor()

    Returns
    -------
    tuple[list[DiscreteVariable], np.ndarray]
        Tuple containing resampled particles and corresponding logged importance weights
    """
    loc_n = len(logw)
    N = loc_n * exec.P

    ncopies = get_number_of_copies(logw.astype('float32'), rng, exec)
    particles = exec.redistribute(particles, ncopies)
    logw = np.log(np.ones(loc_n) / N)

    return particles, logw
