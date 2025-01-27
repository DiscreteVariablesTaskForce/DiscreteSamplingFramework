import math
import sys
import numpy as np
import copy


from discretesampling.base.random import RNG
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.base.executor import Executor
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.domain.gaussian_mixture.mix_model_distribution import decode

from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model

from mpi4py import MPI

def pad(particles):
    mlen = max([i.Gaussian_Mix_Model.k for i in particles])
    return np.array([i.encode(mlen-i.Gaussian_Mix_Model.k) for i in particles])

def restore(coded_particles):
    #print(f'decoding the following array: {coded_particles}')
    return np.array([decode(i) for i in coded_particles])

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


def RJMCMC_Step(particle, grow_prob=[0.5, 0, 0.5]):
    # Propose new samples from the particle front

    proposed_cont = particle.continuous_forward_sample()
    front = proposed_cont.discrete_forward_sample(move_pmf=grow_prob)

    # Compute discrete probabilities

    cont_move = -proposed_cont.continuous_forward_eval(front) + front.continuous_forward_eval(proposed_cont)
    # print(f'cont move = {cont_move}')

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
    # elif front.last_move == 'stick':
    # acc_ratio = proposed_cont.eval()-front.eval() + cont_move
    else:
        acc_ratio = 0

    u = np.random.uniform(0, 1)
    if u < math.exp(acc_ratio):
        return front, True
    else:
        return particle, False


def SMCstep(particle, weight, grow_prob=[0.5, 0, 0.5]):
    # Propose new samples from the particle front

    proposed_cont = particle.continuous_forward_sample()
    front = proposed_cont.discrete_forward_sample(move_pmf=grow_prob)

    # Compute discrete probabilities

    # cont_move = -proposed_cont.continuous_forward_eval(front) + front.continuous_forward_eval(proposed_cont)
    # print(f'cont move = {cont_move}')

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
    # elif front.last_move == 'stick':
    # acc_ratio = proposed_cont.eval()-front.eval() + cont_move
    else:
        acc_ratio = 0

    # print(f'acceptance rate is {acc_ratio}')
    return front, weight + acc_ratio


def one_step_sample_MPI(x_list, y_list):
    print('sample proposition now begins')
    sys.stdout.flush()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: Divide the lists into chunks to distribute over processes
    n = len(x_list)
    chunk_size = n // size
    remainder = n % size

    # Debug print: Check the division of work
    if rank == 0:
        print(f"Total elements: {n}, chunk size: {chunk_size}, remainder: {remainder}")
    sys.stdout.flush()
    # Determine the indices for the portion of the list each process will handle
    if rank < remainder:
        local_start = rank * (chunk_size + 1)
        local_end = local_start + chunk_size + 1
    else:
        local_start = rank * chunk_size + remainder
        local_end = local_start + chunk_size

    # Debug print: Check local range for each rank
    print(f"Rank {rank}: local range {local_start} to {local_end}")

    # Step 2: Scatter the x_list and y_list segments to each process
    local_x = x_list[local_start:local_end]
    local_y = y_list[local_start:local_end]

    # Debug print: Check the local chunks
    #print(f"Rank {rank}: local_x = {local_x}, local_y = {local_y}")

    # Step 3: Each process computes func(x, y) for its segment, which now returns two values
    local_results1 = []
    local_results2 = []

    for x, y in zip(local_x, local_y):

        res1, res2 = SMCstep(x, y)

        local_results1.append(res1)
        local_results2.append(res2)
    # Barrier to ensure all processes have completed before gathering results
    comm.barrier()
    sys.stdout.flush()

    # Step 4: Gather the results from all processes back at the root process
    #print(f"Rank {rank}: Reached comm.gather with local_results1 = {local_results1}, local_results2 = {local_results2}")
    gathered_results1 = comm.gather(local_results1, root=0)
    gathered_results2 = comm.gather(local_results2, root=0)
    comm.barrier()
    #print('gathering')
    sys.stdout.flush()

    if rank == 0:
        # Flatten the gathered results to get the final result in the original order
        result1 = np.array([item for sublist in gathered_results1 for item in sublist])
        result2 = np.array([item for sublist in gathered_results2 for item in sublist])

        # Debug print: Check the gathered results
        #print(f"Rank {rank}: gathered results1 = {result1}, gathered results2 = {result2}")

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
    std = neff / (max(inddict.keys()) + 2)

    allprobs = []
    for i in range(max(inddict.keys()) + 2):
        if i == 0:
            allprobs.append(10 ** -235)
        elif i in inddict.keys():
            allprobs.append(sum([np.exp(weights[j]) for j in inddict[i]]))
        else:
            allprobs.append(std)

    probdicts = {}
    for i in range(1, max(inddict.keys()) + 1):
        ps = np.array(
            [max(10 ** -236, allprobs[i - 1]), max(10 ** -236, allprobs[i]), max(10 ** -236, allprobs[i + 1])])
        probdicts[i] = ps / sum(ps)

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
        a += prop[1]
        print(f'Acceptance rate = {np.round(a / t, 2)}')
        print(f'Current components = {path[-1].Gaussian_Mix_Model.k}')
        print(f'Current BIC = {path[-1].bic()}')

    return path, k, bic


def wt_informed_onestep_MPI(particles, logweights):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # make local PMFs for all values of k available to all nodes
    if rank == 0:
        probdict = get_probdict(particles, logweights)
    else:
        probdict = None

    comm.bcast(probdict, root=0)

    # Divide the lists into chunks to distribute over processes
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
        # print(f'growth probs are {probdict[x.Gaussian_Mix_Model.k]}')
        res1, res2 = SMCstep(x, y, grow_prob=probdict[x.Gaussian_Mix_Model.k])
        local_results1.append(res1)
        local_results2.append(res2)

    # Step 4: Gather the results from all processes back at the root process
    gathered_results1 = comm.gather(local_results1, root=0)
    gathered_results2 = comm.gather(local_results2, root=0)

    if rank == 0:
        # Flatten the gathered results to get the final result in the original order
        result1 = np.array([item for sublist in gathered_results1 for item in sublist])
        result2 = np.array([item for sublist in gathered_results2 for item in sublist])
        return np.array(result1), np.array(result2)
        sys.stdout.flush()
    else:
        return None, None


def straight_SMC_MPI(init, N, T):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    loc_n = int(N)
    # rank = self.exec.rank

    seed = 0
    mvrs_rng = RNG(seed)
        # rngs = [RNG(i + rank * loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle

    initialParticles = [init.get_initial_dist()] * N
    current_particles = initialParticles
    logWeights = np.array([-math.log(N)]*N)

    particle_path = [current_particles]
    logwt_path = [logWeights]

    # display_progress_bar = verbose and rank == 0
    # progress_bar = tqdm(total=Tsmc, desc="SMC sampling", disable=not display_progress_bar)

    sys.stdout.flush()
    #current_particles = comm.bcast(current_particles, root=0)
    #logWeights = comm.bcast(logWeights, root=0)

    #front = comm.bcast(front, root=0)
    #logwts_front = comm.bcast(logwts_front, root=0)

    for t in range(T):

        comm.barrier()
        print(f'beginning step {t}')
        #print(f'Current particles on rank {rank}: {current_particles}')
        sys.stdout.flush()

        logWeights = normalise(logWeights, exec = Executor_MPI())
        #print(f'normalised logWeights: {logWeights}')
        neff = ess(logWeights, exec = Executor_MPI())

        if math.log(neff) < math.log(N) - math.log(2):
            print(f'resampling at step {t} because ESS = {neff}')
            #rint(f'Current_particles at rank {rank}, are {current_particles}')

            current_particles = pad(current_particles)
            current_particles, logWeights = systematic_resampling(
                current_particles, logWeights, mvrs_rng, exec = Executor_MPI())
            #current_particles = comm.bcast(current_particles, root=0)
            current_particles = restore(current_particles)

            #print(f'post-resampling Logweights are {logWeights}')


        new_particles = copy.copy(current_particles)

        sys.stdout.flush()

        # Propose new samples from the particle front in a single step

        #current_particles, logWeights = one_step_sample_MPI(new_particles, logWeights)


        prop_front = []
        prop_wts = []
        for i in range(len(current_particles)):
            i_front, i_wt = SMCstep(new_particles[i], logWeights[i])
            prop_front.append(i_front)
            prop_wts.append(i_wt)

        print(f'Component Lengths: {[i.Gaussian_Mix_Model.k for i in prop_front]}')

        # retain sample

        particle_path.append(np.array(prop_front))
        logwt_path.append(np.array(prop_wts))

        print(f'Path length {len(particle_path)}')

        current_particles = prop_front
        logWeights = prop_wts


    return particle_path, logwt_path


def wt_informed_RJSMC_MPI(init, N, T):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    sys.stdout.flush()
    t = 0
    n = 0

    eff_ss = []

    # Initialise the particles and weights
    current_particles = []
    while n < N:
        current_particles.append(init.get_initial_dist())
        n += 1
    logWeights = np.array([-math.log(N)] * N)
    norm_est = [-np.log(N)]

    particle_path = [current_particles]
    logwt_path = [logWeights]

    print(f'Beginning step {t}')
    while t < T:

        print(f'Beginning step {t}')
        sys.stdout.flush()

        # Extract latest normalising constant estimate
        norm_t = np.log(sum(np.exp(logwts_front))) - math.log(N)
        norm_est.append(norm_t)
        print(f'logZ = {norm_est[-1]}')
        sys.stdout.flush()

        normed_logwts_front = normalise(logwts_front, exec=Executor_MPI())
        sys.stdout.flush()

        # check ESS of sampling front, and resample if necessary
        neff = ess(normed_logwts_front, exec=Executor_MPI())
        eff_ss.append(neff)
        print(f'Effective sample size is {neff}')
        sys.stdout.flush()

        if math.log(neff) < math.log(N) - math.log(2):
            #coded_front = pad(current_particles)
            coded_front, logwts_front = systematic_resampling(
                coded_front, normed_logwts_front, rng=RNG(), exec=Executor_MPI())
            #front = restore(coded_front, current_particles)

        sys.stdout.flush()

        # Propose new samples from the particle front in a single step

        front, logwts_front = wt_informed_onestep_MPI(front, logwts_front)
        front = comm.bcast(front, root=0)
        logwts_front = comm.bcast(logwts_front, root=0)
        print(f'logwts front:{logwts_front}')

        # retain sample
        particle_path.append(front)
        logwt_path.append(logwts_front)
        print(f'weights of the new particles: {logwts_front}')
        sys.stdout.flush()

        t += 1

    return particle_path, logwt_path, eff_ss, norm_est


