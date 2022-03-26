import numpy as np
import random
import math
from scipy.special import logsumexp
from multiprocessing import Process, Manager


class DiscreteVariable:
    def __init__(self):
        pass

    @classmethod
    def getProposalType(self):
        return DiscreteVariableProposal

    @classmethod
    def getTargetType(self):
        return DiscreteVariableTarget

    @classmethod
    def getLKernelType(self):
        # Forward proposal
        return self.getProposalType()

    @classmethod
    def getOptimalLKernelType(self):
        return DiscreteVariableOptimalLKernel


class DiscreteVariableProposal:
    def __init__(self, values, probs):
        # Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p" +\
             " of different lengths"
        probs = np.array(probs)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probs)) < tolerance, "Invalid PMF specified," +\
            " sum of probabilities !~= 1.0"
        assert all(probs > 0), "Invalid PMF specified, all probabilities" +\
            " must be > 0"
        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)

    @classmethod
    def norm(self, x):
        return 1

    @classmethod
    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    # where x and y are norm values from the above function
    def heuristic(self, x, y):
        return True

    def sample(self):
        q = random.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y):
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


# Exact same as proposal above
class DiscreteVariableInitialProposal:
    def __init__(self, values, probs):
        # Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p" +\
            " of different lengths"
        probs = np.array(probs)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probs)) < tolerance, "Invalid PMF specified," +\
            " sum of probabilities !~= 1.0"
        assert all(probs > 0), "Invalid PMF specified, all probabilities" +\
            " must be > 0"

        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)

    def sample(self):
        q = random.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y):
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


class DiscreteVariableTarget:
    def __init__(self):
        pass

    def eval(self, x):
        logprob = -math.inf
        return logprob


class DiscreteVariableOptimalLKernel:
    def __init__(self, current_particles, previous_particles,
                 parallel=False, num_cores=None):
        self.current_particles = current_particles
        self.previous_particles = previous_particles
        self.proposalType = type(self.current_particles[0]).getProposalType()
        self.parallel = parallel
        self.num_cores = num_cores

        self.forward_proposals = [
            self.proposalType(particle) for particle in self.previous_particles
        ]

        # If parallel, calculate eta and proposal_possible in parallel
        # Then precalculate logprob
        if (self.parallel):
            current_particles_split = np.array_split(current_particles,
                                                     self.num_cores)

            with Manager() as manager:
                proposal_possible = manager.list([
                    manager.list(range(len(previous_particles)))
                    for _ in range(len(current_particles))
                ])
                eta = manager.list(range(len(previous_particles)))

                jobs = []
                save_results_in = []  # indexes to save in for each core
                counter = 0
                for i in range(self.num_cores):
                    save_results_in.append([])
                    for p in range(len(current_particles_split[i])):
                        save_results_in[i].append(counter)
                        counter += 1
                for i in range(self.num_cores):  # number of cores to use
                    jobs.append(Process(
                        target=self.get_eta_and_proposal_possible,
                        args=[current_particles_split[i], previous_particles,
                              proposal_possible, eta, save_results_in[i]]
                    ))
                for i in range(len(jobs)):
                    jobs[i]. start()
                for i in range(len(jobs)):
                    jobs[i]. join()

                labels = list(proposal_possible)
                proposal_possible = []
                for label in labels:
                    proposal_possible.append(list(label))

                self.proposal_possible = np.array(proposal_possible)
                self.eta = list(eta)

            # Now precalculate logprob in parallel
            with Manager() as manager:
                logprob = manager.list(range(len(current_particles)))

                jobs = []
                save_results_in = []  # indexes to save in for each core
                counter = 0
                for i in range(self.num_cores):
                    save_results_in.append([])
                    for p in range(len(current_particles_split[i])):
                        save_results_in[i].append(counter)
                        counter += 1
                for i in range(self.num_cores):  # number of cores to use
                    jobs.append(Process(
                        target=self.get_logprob,
                        args=[logprob, save_results_in[i]]
                    ))
                for i in range(len(jobs)):
                    jobs[i]. start()
                for i in range(len(jobs)):
                    jobs[i]. join()
                self.logprob = list(logprob)

        else:
            self.eta = self.calculate_eta(previous_particles)
            self.proposal_possible = self.calculate_proposal_possible(
                previous_particles, current_particles, self.proposalType
            )

            # Precalculate logprob
            self.logprob = self.calculate_logprob(range(len(current_particles)))

    def eval(self, p):
        return self.logprob[p]

    # Calculate logprob for this set of particles
    def get_logprob(self, logprob, save_results_in):
        # One core's worth of log prob calculations
        this_logprob = self.calculate_logprob(save_results_in)

        for i in range(len(save_results_in)):
            logprob[save_results_in[i]] = this_logprob[i]

    def calculate_logprob(self, new_particle_indexes):
        this_logprob = np.full(len(new_particle_indexes), -math.inf)
        for i in range(len(new_particle_indexes)):
            p = new_particle_indexes[i]
            forward_probabilities = np.zeros(len(self.previous_particles))
            for j in range(len(self.previous_particles)):
                if self.proposal_possible[p, j] == 1:
                    forward_probabilities[j] = self.forward_proposals[j].eval(
                        self.current_particles[p]
                    )

            eta_numerator = self.eta[p]
            forward_probability_numerator = forward_probabilities[p]

            numerator = forward_probability_numerator + math.log(eta_numerator)
            denominator_p = np.array([
                forward_probabilities[i] + math.log(self.eta[i])
                for i in range(len(forward_probabilities))
            ])

            denominator = logsumexp(np.setdiff1d(denominator_p, -math.inf))
            this_logprob[i] = numerator - denominator

        return this_logprob

    @staticmethod
    def get_eta_and_proposal_possible(new_particles, current_particles,
                                      proposal_possible, eta, save_results_in):
        # One core's worth of proposal_possible and eta
        this_proposal_possible = DiscreteVariableOptimalLKernel.calculate_proposal_possible(
            current_particles, new_particles,
            type(current_particles[0]).getProposalType()
        )
        this_eta = DiscreteVariableOptimalLKernel.calculate_eta(current_particles)

        for i in range(len(save_results_in)):
            proposal_possible[save_results_in[i]] = this_proposal_possible[i]
            eta[save_results_in[i]] = this_eta[i]

    @staticmethod
    def calculate_eta(particles):
        particles_tmp = list(particles)
        nParticles = len(particles_tmp)
        eta = np.zeros(nParticles)
        for i in range(nParticles):
            eta[i] = particles_tmp.count(particles_tmp[i]) / nParticles

        return eta

    @staticmethod
    def calculate_proposal_possible(previous_particles, current_particles,
                                    proposalType):

        heuristic_function = proposalType.heuristic
        previous_positions = [proposalType.norm(particle) for particle in previous_particles]
        current_positions = [proposalType.norm(particle) for particle in current_particles]
        nPrevious = len(previous_positions)
        nCurrent = len(current_positions)

        proposal_possible = np.zeros([nCurrent, nPrevious])

        for i in range(nCurrent):
            for j in range(nPrevious):
                proposal_possible[i, j] = int(heuristic_function(
                    current_positions[i], previous_positions[j]
                ))
        return proposal_possible
