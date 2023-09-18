import copy
import numpy as np
import math
from discretesampling.base.random import RNG
from discretesampling.base.executor.executor import Executor
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling


class DiscreteVariableSMC():

    def __init__(self, variableType, target, initialProposal,
                 use_optimal_L=False,
                 exec=Executor()):
        """_summary_

        Parameters
        ----------
        variableType : _type_
            _description_
        target : _type_
            _description_
        initialProposal : _type_
            _description_
        use_optimal_L : bool, optional
            _description_, by default False
        exec : _type_, optional
            _description_, by default Executor()
        """        """"""
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.use_optimal_L = use_optimal_L
        self.exec = exec

        if use_optimal_L:
            self.LKernelType = variableType.getOptimalLKernelType()
        else:
            # By default getLKernelType just returns
            # variableType.getProposalType(), the same as the forward_proposal
            self.LKernelType = variableType.getLKernelType()

        self.initialProposal = initialProposal
        self.target = target

    def sample(self, Tsmc: int, N: int, seed: int = 0):
        """Generate samples from the SMC sampler.

        Parameters
        ----------
        Tsmc : int
            Number of iterations to run SMC sampler for.
        N : int
            Number of particles
        seed : int, optional
            Random seed, by default 0

        Returns
        -------
        : list[DiscreteVariable]
            List of generated samples of type specified in constructor.
        """        """"""
        loc_n = int(N/self.exec.P)
        rank = self.exec.rank
        mvrs_rng = RNG(seed)
        rngs = [RNG(i + rank*loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle

        initialParticles = [self.initialProposal.sample(rngs[i], self.target) for i in range(loc_n)]
        current_particles = initialParticles
        logWeights = np.array([self.target.eval(p) - self.initialProposal.eval(p, self.target) for p in initialParticles])

        for t in range(Tsmc):
            logWeights = normalise(logWeights, self.exec)
            neff = ess(logWeights, self.exec)

            if math.log(neff) < math.log(N) - math.log(2):
                current_particles, logWeights = systematic_resampling(
                    current_particles, logWeights, mvrs_rng, exec=self.exec)

            new_particles = copy.copy(current_particles)

            forward_logprob = np.zeros(len(current_particles))

            # Sample new particles and calculate forward probabilities
            for i in range(loc_n):
                forward_proposal = self.proposalType(current_particles[i], rng=rngs[i])
                new_particles[i] = forward_proposal.sample()
                forward_logprob[i] = forward_proposal.eval(new_particles[i])

            if self.use_optimal_L:
                Lkernel = self.LKernelType(
                    new_particles, current_particles, parallel=self.exec, num_cores=1
                )
            for i in range(loc_n):
                if self.use_optimal_L:
                    reverse_logprob = Lkernel.eval(i)
                else:
                    Lkernel = self.LKernelType(new_particles[i])
                    reverse_logprob = Lkernel.eval(current_particles[i])

                current_target_logprob = self.target.eval(current_particles[i])
                new_target_logprob = self.target.eval(new_particles[i])

                logWeights[i] += new_target_logprob - current_target_logprob + reverse_logprob - forward_logprob[i]

            current_particles = new_particles

        return current_particles
