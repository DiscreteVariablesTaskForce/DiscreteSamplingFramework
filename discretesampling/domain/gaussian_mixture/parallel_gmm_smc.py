import math
import numpy as np
import joblib
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from mpi4py import MPI

from discretesampling.base.random import RNG
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling
from discretesampling.base.algorithms.smc_components.normalisation import normalise

from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model

from mpi4py import MPI

def read_floats_from_file(filepath):
    floats = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                floateval = [float(value) for value in line.split()]
                floats.extend(floateval)
            except ValueError:
                print(f'Non-float detected in line {line}')

    return floats

def RJMCMC_Step(particle, grow_prob=[0.5,0,0.5]):
    # Propose new samples from the particle front

    proposed_cont = particle.continuous_forward_sample()
    front = proposed_cont.discrete_forward_sample(move_pmf = grow_prob)

    # Compute discrete probabilities

    cont_move = -proposed_cont.continuous_forward_eval(front) + front.continuous_forward_eval(proposed_cont)
    #print(f'cont move = {cont_move}')

    if front.last_move == 'split':
        fwd = front.split_log_eval(proposed_cont, grow_prob[2])
        back = proposed_cont.merge_log_eval(front, grow_prob[0])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'merge':
        fwd = front.merge_log_eval(proposed_cont, grow_prob[0])
        back = proposed_cont.split_log_eval(front, grow_prob[2])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'birth':
        fwd = front.birth_log_eval(proposed_cont, grow_prob[2])
        back = proposed_cont.death_log_eval(front, grow_prob[0])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'death':
        fwd = front.death_log_eval(proposed_cont, grow_prob[0])
        back = proposed_cont.birth_log_eval(front, grow_prob[2])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'stick':
        acc_ratio = proposed_cont.eval()-front.eval() + cont_move
    else:
        acc_ratio = 0

    u = np.random.uniform(0,1)
    if u < math.exp(acc_ratio):
        return front, True
    else:
        return particle, False

def SMCstep(particle, weight, grow_prob=[0.5,0,0.5]):
    # Propose new samples from the particle front

    proposed_cont = particle.continuous_forward_sample()
    front = proposed_cont.discrete_forward_sample(move_pmf = grow_prob)

    # Compute discrete probabilities

    cont_move = -proposed_cont.continuous_forward_eval(front) + front.continuous_forward_eval(proposed_cont)
    #print(f'cont move = {cont_move}')

    if front.last_move == 'split':
        fwd = front.split_log_eval(proposed_cont, grow_prob[2])
        back = proposed_cont.merge_log_eval(front, grow_prob[0])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'merge':
        fwd = front.merge_log_eval(proposed_cont, grow_prob[0])
        back = proposed_cont.split_log_eval(front, grow_prob[2])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'birth':
        fwd = front.birth_log_eval(proposed_cont, grow_prob[2])
        back = proposed_cont.death_log_eval(front, grow_prob[0])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'death':
        fwd = front.death_log_eval(proposed_cont, grow_prob[0])
        back = proposed_cont.birth_log_eval(front, grow_prob[2])
        acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
    elif front.last_move == 'stick':
        acc_ratio = proposed_cont.eval()-front.eval() + cont_move
    else:
        acc_ratio = 0

    #print(f'acceptance rate is {acc_ratio}')
    return front, weight+acc_ratio

def one_step_sample_MPI(x_list,y_list):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: Divide the lists into chunks to distribute over processes
    n = len(x_list)
    chunk_size = n // size
    remainder = n % size

    # Determine the indices for the portion of the list each process will handle
    if rank < remainder:
        local_start = rank * (chunk_size + 1)
        local_end = local_start + chunk_size + 1
    else:
        local_start = rank * chunk_size + remainder
        local_end = local_start + chunk_size

    # Step 2: Scatter the x_list and y_list segments to each process
    local_x = x_list[local_start:local_end]
    local_y = y_list[local_start:local_end]

    # Step 3: Each process computes func(x, y) for its segment, which now returns two values
    local_results1 = []
    local_results2 = []

    for x, y in zip(local_x, local_y):
        res1, res2 = SMCstep(x, y)
        local_results1.append(res1)
        local_results2.append(res2)

    # Step 4: Gather the results from all processes back at the root process
    gathered_results1 = comm.gather(local_results1, root=0)
    gathered_results2 = comm.gather(local_results2, root=0)

    if rank == 0:
        # Flatten the gathered results to get the final result in the original order
        result1 = [item for sublist in gathered_results1 for item in sublist]
        result2 = [item for sublist in gathered_results2 for item in sublist]
        return result1, result2
    else:
        return None, None



def get_k_indices(particles):
    inddict = {}
    for i in particles:
        if i.Gaussian_Mix_Model.k not in inddict.keys():
            inddict[i.Gaussian_Mix_Model.k] = [list(particles).index(i)]
        else:
            inddict[i.Gaussian_Mix_Model.k].append(list(particles).index(i))

    return dict(sorted(inddict.items()))

def get_probdict(particles, weights):
    neff = ess(weights, exec=Executor_MPI())
    inddict = get_k_indices(particles)
    std = neff/(max(inddict.keys()) + 2)

    allprobs = []
    for i in range(max(inddict.keys())+2):
        if i == 0:
            allprobs.append(10**-235)
        elif i in inddict.keys():
            allprobs.append(sum([np.exp(weights[j]) for j in inddict[i]]))
        else:
            allprobs.append(std)

    probdicts = {}
    for i in range(1, max(inddict.keys())+1):
        ps = np.array([max(10**-236,allprobs[i-1]), max(10**-236,allprobs[i]), max(10**-236,allprobs[i+1])])
        probdicts[i] = ps/sum(ps)

    print(f'jump probabilities are {probdicts}')

    return probdicts

def RJMCMC(init, T):
    k = []
    bic = []
    path = [init.get_initial_dist()]

    t = 0
    a = 0
    while t < T:
        prop = RJMCMC_Step(path[-1])
        path.append(prop[0])
        k.append(prop[0].Gaussian_Mix_Model.k)
        bic.append(prop[0].bic())
        print(f'Completed step {t}')
        t += 1
        a+=prop[1]
        print(f'Acceptance rate = {np.round(a/t, 2)}')
        print(f'Current components = {path[-1].Gaussian_Mix_Model.k}')
        print(f'Current BIC = {path[-1].bic()}')


    return path, k, bic

def wt_informed_onestep_MPI(particles, logweights):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #make local PMFs for all values of k available to all nodes
    if rank == 0:
        probdict = get_probdict(particles,logweights)
    else:
        probdict = None

    comm.bcast(probdict, root=0)

    #Divide the lists into chunks to distribute over processes
    n = len(particles)
    chunk_size = n // size
    remainder = n % size

    # Determine the indices for the portion of the list each process will handle
    if rank < remainder:
        local_start = rank * (chunk_size + 1)
        local_end = local_start + chunk_size + 1
    else:
        local_start = rank * chunk_size + remainder
        local_end = local_start + chunk_size

    # Step 2: Scatter the x_list and y_list segments to each process
    local_parts = particles[local_start:local_end]
    local_logwts = logweights[local_start:local_end]

    # Step 3: Each process computes func(x, y) for its segment, which now returns two values
    local_results1 = []
    local_results2 = []

    for x, y in zip(local_parts, local_logwts):
        #print(f'growth probs are {probdict[x.Gaussian_Mix_Model.k]}')
        res1, res2 = SMCstep(x, y, grow_prob = probdict[x.Gaussian_Mix_Model.k])
        local_results1.append(res1)
        local_results2.append(res2)

    # Step 4: Gather the results from all processes back at the root process
    gathered_results1 = comm.gather(local_results1, root=0)
    gathered_results2 = comm.gather(local_results2, root=0)

    if rank == 0:
        # Flatten the gathered results to get the final result in the original order
        result1 = [item for sublist in gathered_results1 for item in sublist]
        result2 = [item for sublist in gathered_results2 for item in sublist]
        return result1, result2
    else:
        return None, None
def parallel_logsumexp(log_probs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # Total number of processes
    rank = comm.Get_rank()  # Rank of this process

    # Step 1: Scatter the data to different processes
    n = len(log_probs)
    local_n = n // size  # Size of each chunk

    # Scatter the data (handling case where data is not divisible by size)
    local_data = comm.scatter([log_probs[i::size] for i in range(size)], root=0)

    # Step 2: Find local maximum log-probability
    local_max = max(local_data)

    # Step 3: Find global maximum log-probability
    global_max = comm.allreduce(local_max, op=MPI.MAX)

    # Step 4: Compute local sum of exp(log_prob - global_max) for numerical stability
    local_sum_exp = sum(math.exp(x - global_max) for x in local_data)

    # Step 5: Compute global sum of exp(log_prob - global_max)
    global_sum_exp = comm.allreduce(local_sum_exp, op=MPI.SUM)

    # Step 6: Compute final logsumexp
    logsumexp = global_max + math.log(global_sum_exp)

    return logsumexp

def parallel_ess(logw):
    """
    Description
    -----------
    Computes the Effective Sample Size of the given normalised weights

    Parameters
    ----------
    logw : array of logged importance normalised weights

    Returns
    -------
    double scalar : Effective Sample Size

    """

    mask = np.invert(np.isneginf(logw))  # mask to filter out any weight = 0 (or -inf in log-scale)

    logw = np.array([logw[i] for i in range(len(logw)) if mask[i]])
    inverse_neff = np.exp(parallel_logsumexp(2*logw))

    return 1 / inverse_neff

def normalize_log_probs(log_probs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Step 1: Compute the logsumexp of the log-probabilities
    logsumexp_val = parallel_logsumexp(log_probs)

    # Step 2: Normalize the log-probabilities by subtracting logsumexp
    # Scatter the log-probs again
    local_data = comm.scatter([log_probs[i::comm.Get_size()] for i in range(comm.Get_size())], root=0)

    # Each process normalizes its own chunk of data
    local_normalized = [log_prob - logsumexp_val for log_prob in local_data]

    # Gather normalized log-probs back at root
    normalized_log_probs = comm.gather(local_normalized, root=0)

    if rank == 0:
        # Flatten the list of normalized log probabilities
        normalized_log_probs = [item for sublist in normalized_log_probs for item in sublist]
        return normalized_log_probs
    else:
        return None


def parallel_systematic_resampling(particles, log_weights):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # Number of processes
    rank = comm.Get_rank()  # Rank of this process

    N = len(particles)  # Total number of particles

    # Broadcast particles and log_weights to all processes
    particles = comm.bcast(particles, root=0)
    log_weights = comm.bcast(log_weights, root=0)

    # Step 1: Convert log-weights to normalized weights using the log-sum-exp trick
    max_log_weight = np.max(log_weights)  # For numerical stability
    weights = np.exp(log_weights - max_log_weight)  # Exponentiate log-weights
    weights /= np.sum(weights)  # Normalize to ensure they sum to 1

    # Step 2: Compute the cumulative sum of the normalized weights
    cdf = np.cumsum(weights)
    print(f'CDF = {cdf}')

    # Ensure the last value in the CDF is exactly 1 (if numerical errors cause small deviations)
    cdf[-1] = 1.0

    # Step 3: Perform systematic resampling
    u0 = np.random.uniform(0, 1 / N)  # Start point for systematic resampling
    u = np.linspace(u0, 1 - 1 / N, N)  # Create N evenly spaced points

    # Step 4: Resample indices based on the CDF
    resampled_indices = np.searchsorted(cdf, u)

    # Fix any out-of-bounds indices by clamping to valid range [0, N-1]
    resampled_indices = np.clip(resampled_indices, 0, N - 1)

    # Resample particles based on the global indices
    resampled_particles = [particles[i] for i in resampled_indices]
    resampled_weights = [-N] * N  # After resampling, weights are uniform

    # Gather resampled particles and weights at the root process
    all_resampled_particles = comm.gather(resampled_particles, root=0)
    all_resampled_weights = comm.gather(resampled_weights, root=0)

    if rank == 0:
        # Flatten the list of resampled particles and weights
        all_resampled_particles = [item for sublist in all_resampled_particles for item in sublist]
        all_resampled_weights = [item for sublist in all_resampled_weights for item in sublist]
        return all_resampled_particles, all_resampled_weights
    else:
        return None, None

def straight_SMC_MPI(init, N, T):
    t = 0
    n = 0

    eff_ss = []

    # Initialise the particles and weights
    front = []
    while n < N:
        front.append(init.get_initial_dist())
        n += 1
    logwts_front = np.array([-math.log(N)] * N)
    norm_est = [-np.log(N)]

    particle_path = [front]
    logwt_path = [logwts_front]

    print(f'Beginning step {t}')
    while t < T:

        print(f'Beginning step {t}')

        # Extract latest normalising constant estimate
        norm_t = logsumexp(logwts_front) - np.log(N)
        norm_est.append(norm_est[-1] + norm_t)
        print(f'Z={norm_est[-1]}')

        # check ESS of sampling front, and resample if necessary
        neff = ess(logwts_front, exec=Executor_MPI())
        eff_ss.append(neff)
        print(f'Effective sample size is {neff}')

        if math.log(neff) < math.log(N) - math.log(2):
            print('Resampling')
            front, logwts_front = systematic_resampling(
                front, logwts_front, rng = RNG(), exec=Executor_MPI())

        # Propose new samples from the particle front in a single step

        front, logwts_front = one_step_sample_MPI(front, logwts_front)



        # Update and normalise weights

        logwts_front = normalise(logwts_front, exec=Executor_MPI())

        # retain sample
        particle_path.append(front)
        logwt_path.append(logwts_front)

        t += 1

    return particle_path, logwt_path, eff_ss, norm_est

def wt_informed_RJSMC_MPI(init,N,T):
    t = 0
    n = 0

    eff_ss = []

    # Initialise the particles and weights
    front = []
    while n < N:
        front.append(init.get_initial_dist())
        n += 1
    logwts_front = np.array([-math.log(N)] * N)
    norm_est = [-np.log(N)]

    particle_path = [front]
    logwt_path = [logwts_front]

    print(f'Beginning step {t}')
    while t < T:

        print(f'Beginning step {t}')

        # check ESS of sampling front, and resample if necessary
        neff = ess(logwts_front, exec=Executor_MPI())
        eff_ss.append(neff)
        print(f'Effective sample size is {neff}')

        if math.log(neff) < math.log(N) - math.log(2):
            print('Resampling')
            front, logwts_front = systematic_resampling(
                front, logwts_front, rng=RNG(), exec=Executor_MPI())

        # Propose new samples from the particle front in a single step

        front, logwts_front = wt_informed_onestep_MPI(front, logwts_front)

        # Extract latest normalising constant estimate
        norm_t = logsumexp(logwts_front) - np.log(N)
        norm_est.append(norm_est[-1] + norm_t)
        print(f'logZ = {norm_est[-1]}')

        # Update and normalise weights
        logwts_front = normalise(logwts_front, exec=Executor_MPI())

        # retain sample
        particle_path.append(front)
        logwt_path.append(logwts_front)

        t += 1

    return particle_path, logwt_path, eff_ss, norm_est

