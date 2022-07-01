import multiprocessing
from .smc import DiscreteVariableSMC
import copy
import numpy as np


class DiscreteVariableSMC_MP(DiscreteVariableSMC):
    def evolve(self, particles):
        P = len(particles)
        new_particles = copy.deepcopy(particles)
        forward_logprob = np.zeros(P)

        with multiprocessing.Pool() as pool:
            new_particles, forward_logrob = zip(*pool.map(super().evolve_particle, particles))

        return new_particles, forward_logprob
