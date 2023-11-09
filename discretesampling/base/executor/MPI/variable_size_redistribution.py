from discretesampling.base.util import pad, restore
from smccomponents.resample.mpi.redistribution import (
    fixed_size_redistribution
)


def variable_size_redistribution(particles, ncopies, exec):
    x = pad(particles, exec)

    x = fixed_size_redistribution(x, ncopies)

    particles = restore(x, particles)

    return particles
