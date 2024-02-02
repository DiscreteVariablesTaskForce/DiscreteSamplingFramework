import copy
import numpy as np
import math
from tqdm.auto import tqdm
from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling
#from discretesampling.base.algorithms.smc_components.systematic_resampling import systematic_resampling
#from discretesampling.base.algorithms.smc_components.importance_resampling import importance_resampling


class DiscreteVariableSMC():

    def __init__(self, variableType, target, initialProposal, proposal=None,
                 Lkernel=None,
                 use_optimal_L=False,
                 exec=Executor()):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.proposal = proposal
        if proposal is None:
            self.proposal = self.proposalType()
        self.use_optimal_L = use_optimal_L
        self.exec = exec

        if use_optimal_L:
            self.LKernelType = variableType.getOptimalLKernelType()
        else:
            # By default getLKernelType just returns
            # variableType.getProposalType(), the same as the forward_proposal
            self.LKernelType = variableType.getLKernelType()

        self.Lkernel = Lkernel
        if Lkernel is None:
            self.Lkernel = self.LKernelType()

        self.initialProposal = initialProposal
        self.target = target

    def sample(self, Tsmc, N, seed=0, verbose=True):
        loc_n = int(N/self.exec.P)
        rank = self.exec.rank
        mvrs_rng = RNG(seed)
        rngs = [RNG(i + rank*loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle

        initialParticles = [self.initialProposal.sample(rngs[i], self.target) for i in range(loc_n)]
        current_particles = initialParticles
        logWeights = np.array([self.target.eval(p) - self.initialProposal.eval(p, self.target) for p in initialParticles])

        display_progress_bar = verbose and rank == 0
        progress_bar = tqdm(total=Tsmc, desc="SMC sampling", disable=not display_progress_bar)

        for t in range(Tsmc):
            logWeights = normalise(logWeights, self.exec)
            neff = ess(logWeights, self.exec)

            if math.log(neff) < math.log(N) - math.log(2):
                 current_particles, logWeights = systematic_resampling(
                       current_particles, logWeights, mvrs_rng, exec=self.exec)
                 # current_particles, logWeights = systematic_resampling_log(
                 #      current_particles, np.exp(logWeights), mvrs_rng)
                 #   current_particles, logWeights = importance_resampling(
                 #       current_particles, np.exp(logWeights), mvrs_rng)


            new_particles = copy.copy(current_particles)

            forward_logprob = np.zeros(len(current_particles))

            # Sample new particles and calculate forward probabilities
            for i in range(loc_n):
                forward_proposal = self.proposal
                new_particles[i] = forward_proposal.sample(current_particles[i], rng=rngs[i])
                forward_logprob[i] = forward_proposal.eval(current_particles[i], new_particles[i])

            if self.use_optimal_L:
                Lkernel = self.LKernelType(
                    new_particles, current_particles, parallel=self.exec, num_cores=1
                )
            for i in range(loc_n):
                if self.use_optimal_L:
                    reverse_logprob = Lkernel.eval(i)
                else:
                    Lkernel = self.Lkernel
                    reverse_logprob = Lkernel.eval(new_particles[i], current_particles[i])

                current_target_logprob = self.target.eval(current_particles[i])
                new_target_logprob = self.target.eval(new_particles[i])

                logWeights[i] += new_target_logprob - current_target_logprob + reverse_logprob - forward_logprob[i]

            current_particles = new_particles
            progress_bar.update(1)

        progress_bar.close()
        return current_particles
