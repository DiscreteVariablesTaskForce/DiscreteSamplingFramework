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
from discretesampling.base.util import pad, restore
from discretesampling.domain.gaussian_mixture.mix_model_distribution import decode

from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model

from mpi4py import MPI

'''
def pad(particles):
    mlen = max([i.Gaussian_Mix_Model.k for i in particles])
    return np.array([i.encode(mlen-i.Gaussian_Mix_Model.k) for i in particles])

def restore(coded_particles):
    #print(f'decoding the following array: {coded_particles}')
    return np.array([decode(i) for i in coded_particles])
'''

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


def one_step_sample_MPI(particles, weights):
    sys.stdout.flush()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    '''
    # Step 1: Divide the lists into chunks to distribute over processes
    n = len(x_list)
    chunk_size = n // size
    remainder = n % size


    sys.stdout.flush()
    # Determine the indices for the portion of the list each process will handle
    if rank < remainder:
        local_start = rank * (chunk_size + 1)
        local_end = local_start + chunk_size + 1
    else:
        local_start = rank * chunk_size + remainder
        local_end = local_start + chunk_size

    # Debug print: Check local range for each rank


    # Step 2: Scatter the x_list and y_list segments to each process
    local_x = x_list[local_start:local_end]
    local_y = y_list[local_start:local_end]


    # Step 3: Each process computes func(x, y) for its segment, which now returns two values
    
    '''
    local_results1 = []
    local_results2 = []
    for x, y in zip(particles, weights):

        res1, res2 = SMCstep(x, y)

        local_results1.append(res1)
        local_results2.append(res2)

    # Barrier to ensure all processes have completed before gathering results
    comm.barrier()
    #print(f'local results on rank {rank}: {local_results2}')
    sys.stdout.flush()

    # Step 4: Gather the results from all processes back at the root process
    gathered_results1 = comm.gather(local_results1, root=0)
    gathered_results2 = comm.gather(local_results2, root=0)
    comm.barrier()
    sys.stdout.flush()

    '''
    if rank == 0:
        # Flatten the gathered results to get the final result in the original order
        result1 = np.array([item for sublist in gathered_results1 for item in sublist])
        result2 = np.array([item for sublist in gathered_results2 for item in sublist])


        return result1, result2
    else:
        return None, None
    '''

    return local_results1, local_results2


def get_k_indices(particles):
    inddict = {}
    for i in particles:
        if i.Gaussian_Mix_Model.k not in inddict.keys():
            inddict[i.Gaussian_Mix_Model.k] = [list(particles).index(i)]
        else:
            inddict[i.Gaussian_Mix_Model.k].append(list(particles).index(i))

    return dict(sorted(inddict.items()))


def get_probdict(particles, weights, neff):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        #print('Computing ess')
        #sys.stdout.flush()
        #neff = ess(weights, exec=Executor_MPI())
        #sys.stdout.flush()
        #print(f'ess = {neff}')
        #comm.barrier()
        #sys.stdout.flush()

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
            ps = np.array([max(10 ** -236, allprobs[i - 1]), max(10 ** -236, allprobs[i]), max(10 ** -236, allprobs[i + 1])])
            probdicts[i] = ps / sum(ps)

        #print(f'jump probabilities are {probdicts}')

        return probdicts

    else:
        return None


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


def wt_informed_onestep_MPI(particles, logweights, neff):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # make local PMFs for all values of k available to all nodes


    comm.barrier()
    sys.stdout.flush()

    all_ps = np.array(comm.gather(particles, root = 0)).ravel()
    all_wts = np.array(comm.gather(logweights, root = 0)).ravel()
    probdict = None

    if rank == 0:
        #print('getting the probability dictionary')
        sys.stdout.flush()
        probdict = get_probdict(all_ps, all_wts, neff)

    probdict = comm.bcast(probdict, root = 0)

    #print(f'rank {rank}: probdict = {probdict}')
    # Step 3: Each process computes func(x, y) for its segment, which now returns two values
    local_results1 = []
    local_results2 = []

    for x, y in zip(particles, logweights):
        res1, res2 = SMCstep(x, y, grow_prob=probdict[x.Gaussian_Mix_Model.k])
        local_results1.append(res1)
        local_results2.append(res2)

    return local_results1, local_results2

def straight_SMC_MPI(init, N, T):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    loc_n = int(N)
    # rank = self.exec.rank

    seed = 0
    mvrs_rng = RNG(seed)
        # rngs = [RNG(i + rank * loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle

    current_particles = [init.get_initial_dist()] * int(N/size)

    logWeights = np.array([-math.log(N)]*int(N/size))
    #print(f'rank = {rank}, weights = {logWeights}')

    all_particles = np.array(comm.gather(current_particles, root=0)).ravel()
    all_wts = np.array(comm.gather(logWeights, root=0)).ravel()

    if rank == 0:
        particle_path = [all_particles]
        logwt_path = [all_wts]
        ess_path = [N]

    sys.stdout.flush()

    for t in range(T):

        comm.barrier()
        if rank == 0:
            print(f'beginning step {t}')
        sys.stdout.flush()

        logWeights = normalise(logWeights, exec = Executor_MPI())
        neff = ess(logWeights, exec = Executor_MPI())
        print(f'rank {rank}: neff = {neff}')

        if rank == 0:
            ess_path.append(neff)

        if math.log(neff) < math.log(N) - math.log(2):

            current_particles = pad(current_particles, exec = Executor_MPI())
            current_particles, logWeights = systematic_resampling(
                current_particles, logWeights, mvrs_rng, exec = Executor_MPI())
            comm.barrier()


            current_particles = restore(current_particles)

        sys.stdout.flush()

        current_particles, logWeights = one_step_sample_MPI(current_particles, logWeights)
        #print(f'rank = {rank}, after sample weights = {logWeights}')



        all_particles = np.array(comm.gather(current_particles, root = 0)).ravel()
        all_wts = np.array(comm.gather(logWeights, root = 0)).ravel()

        if rank == 0:
            particle_path.append(np.array(all_particles))
            logwt_path.append(np.array(all_wts))

        t+=1


    return particle_path, logwt_path, ess_path


def wt_informed_RJSMC_MPI(init, N, T):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    loc_n = int(N)
    # rank = self.exec.rank

    seed = 0
    mvrs_rng = RNG(seed)
    # rngs = [RNG(i + rank * loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle

    current_particles = [init.get_initial_dist()] * int(N / size)

    logWeights = np.array([-math.log(N)] * int(N / size))
    # print(f'rank = {rank}, weights = {logWeights}')


    all_particles = np.array(comm.gather(current_particles, root=0)).ravel()
    all_wts = np.array(comm.gather(logWeights, root=0)).ravel()
    sys.stdout.flush()

    if rank == 0:
        #print(f'gathered weights: {all_wts}')
        particle_path = [all_particles]
        logwt_path = [all_wts]
        ess_path = [N]

    comm.barrier()
    #print(f'logWeights at rank {rank} after scattering:{logWeights}')

    sys.stdout.flush()

    for t in range(T):

        comm.barrier()
        if rank == 0:
            print(f'beginning step {t}')
            sys.stdout.flush()

        logWeights = normalise(logWeights, exec=Executor_MPI())
        #print(f'rank = {rank}, normalised logWeights = {logWeights}')
        sys.stdout.flush()
        # check ESS of sampling front, and resample if necessary
        neff = ess(logWeights, exec=Executor_MPI())
        if rank == 0:
            ess_path.append(neff)
            #print(f'Effective sample size is {neff}')

        sys.stdout.flush()

        if math.log(neff) < math.log(N) - math.log(2):
            current_particles = pad(current_particles, exec=Executor_MPI())
            current_particles, logWeights = systematic_resampling(
                current_particles, logWeights, mvrs_rng, exec=Executor_MPI())
            comm.barrier()

            current_particles = restore(current_particles)

        sys.stdout.flush()

        # Propose new samples from the particle front in a single step

        sys.stdout.flush()

        #prop_front = comm.bcast(prop_front, root=0)
        #prop_wts = comm.bcast(prop_wts, root=0)

        # retain sample
        #print('Doing new sample')
        current_particles, logWeights = wt_informed_onestep_MPI(current_particles, logWeights, neff)
        sys.stdout.flush()
        all_particles = np.array(comm.gather(current_particles, root=0)).ravel()
        all_Weights = np.array(comm.gather(logWeights, root=0)).ravel()

        if rank == 0:
            particle_path.append(all_particles)
            logwt_path.append(all_Weights)

        #print(f'logWeights at rank {rank}:{logWeights}')
        comm.barrier()

        sys.stdout.flush()

        t += 1
    if rank == 0:
        return particle_path, logwt_path, ess_path
    else:
        return None, None, None


