import math
import sys
import numpy as np
import copy
import random


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

def get_wt_average_k(parts, wts):
    normwts = np.exp(normalise(np.array(wts)))
    ks = np.array([i.Gaussian_Mix_Model.k for i in parts])
    return np.dot(normwts, ks)

def get_wt_av_bic(parts, wts):
    normwts = np.exp(normalise(np.array(wts)))
    bics = np.array([i.bic() for i in parts])
    return np.dot(normwts, bics)

def RJMCMC_Step(particle, grow_prob=[0.5, 0, 0.5]):
    # Propose new samples from the particle front

    proposed_cont = particle.continuous_forward_sample()
    front = proposed_cont.discrete_forward_sample(move_pmf=grow_prob)

    # Compute discrete probabilities

    #cont_move = -proposed_cont.continuous_forward_eval(front) + front.continuous_forward_eval(proposed_cont)
    # print(f'cont move = {cont_move}')

    if front.last_move == 'split':
        fwd = front.split_log_eval(proposed_cont, grow_prob[0])
        back = proposed_cont.merge_log_eval(front, grow_prob[2])
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'merge':
        fwd = front.merge_log_eval(proposed_cont, grow_prob[2])
        back = proposed_cont.split_log_eval(front, grow_prob[0])
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'birth':
        fwd = front.birth_log_eval(proposed_cont, grow_prob[0])
        back = proposed_cont.death_log_eval(front, grow_prob[2])
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'death':
        fwd = front.death_log_eval(proposed_cont, grow_prob[2])
        back = proposed_cont.birth_log_eval(front, grow_prob[0])
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    # elif front.last_move == 'stick':
    # acc_ratio = proposed_cont.eval()-front.eval() + cont_move
    else:
        acc_ratio = 0 #front.eval() - particle.eval()

    acc_prob = min(acc_ratio, 0)
    u = np.random.uniform(0, 1)
    if u < math.exp(acc_prob):
        return front, True
    else:
        return particle, False

def nRJMCMC_Step(particle, nu=1):
    # Propose new samples from the particle front
    print(f'Current nu: {nu}')
    proposed_cont = particle.continuous_forward_sample()
    if nu == 1:
        front = proposed_cont.discrete_forward_sample(move_pmf=[0,0,1])
    elif nu == -1:
        front = proposed_cont.discrete_forward_sample(move_pmf=[1,0,0])
    else:
        front = proposed_cont

    # Compute discrete probabilities

    #cont_move = -proposed_cont.continuous_forward_eval(front) + front.continuous_forward_eval(proposed_cont)
    # print(f'cont move = {cont_move}')

    if front.last_move == 'split':
        fwd = front.split_log_eval(proposed_cont, 1)
        back = proposed_cont.merge_log_eval(front, 1)
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'merge':
        fwd = front.merge_log_eval(proposed_cont, 1)
        back = proposed_cont.split_log_eval(front, 1)
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'birth':
        fwd = front.birth_log_eval(proposed_cont, 1)
        back = proposed_cont.death_log_eval(front, 1)
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'death':
        fwd = front.death_log_eval(proposed_cont, 1)
        back = proposed_cont.birth_log_eval(front, 1)
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    # elif front.last_move == 'stick':
    # acc_ratio = proposed_cont.eval()-front.eval() + cont_move
    else:
        acc_ratio = 0 #front.eval() - particle.eval()

    acc_prob = min(acc_ratio, 0)
    u = np.random.uniform(0, 1)
    if u < math.exp(acc_prob):
        return front, True, nu
    else:
        return particle, False, -nu

def get_probdict(particles, weights, neff):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        essprob = 1-(neff/len(particles))
        print(f'neff = {neff/len(particles)}')
        weights = normalise(weights)
        #sys.stdout.flush()
        #neff = ess(weights, exec=Executor_MPI())
        #sys.stdout.flush()
        #print(f'ess = {neff}')
        #comm.barrier()
        #sys.stdout.flush()

        partsort = get_k_indices(particles)
        inddict = partsort[0]
        print(f'dictionary keys are {inddict.keys()}')
        absent = partsort[1]
        print(f'absent = {absent}')
        if absent != 0:
            std = essprob/absent
        else:
            std = 0

        allprobs = []
        for i in inddict:
            if i == 0:
                allprobs.append(0)
            elif inddict[i] == []:
                allprobs.append(std)
            else :
                allprobs.append((neff/len(particles))*sum([np.exp(weights[j]) for j in inddict[i]]))

        allprobs = allprobs/sum(allprobs)
        print(f'PMF is  {allprobs}')
        probdicts = {}
        for i in range(1, max(inddict.keys())):
            ps = np.array([max(10**-235, allprobs[i - 1]), max(10**-235, allprobs[i]), max(10**-235, allprobs[i + 1])])
            probdicts[i] = ps / sum(ps)

        return probdicts

    else:
        return None

def SMCstep(particle, weight, adj_probs = [[1/3, 1/3, 1/3], [1/3,1/3,1/3], [1/3,1/3,1/3]]):
    #print(particle.last_move)
    # Propose new samples from the particle front

    proposed_cont = particle.continuous_forward_sample()
    front = proposed_cont.discrete_forward_sample(move_pmf=adj_probs[1])

    # Compute discrete probabilities

    # cont_move = -proposed_cont.continuous_forward_eval(front) + front.continuous_forward_eval(proposed_cont)
    # print(f'cont move = {cont_move}')

    if front.last_move == 'split':
        fwd = front.split_log_eval(proposed_cont, adj_probs[1][2])
        back = proposed_cont.merge_log_eval(front, adj_probs[2][0])
        #print(f'moves: {fwd[0] + fwd[1]}, {back[1] + back[0]}')
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'merge':
        fwd = front.merge_log_eval(proposed_cont, adj_probs[0][2])
        back = proposed_cont.split_log_eval(front, adj_probs[1][0])
        #print(f'moves: {fwd[0] + fwd[1]}, {back[1] + back[0]}')
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'birth':
        fwd = front.birth_log_eval(proposed_cont, adj_probs[2][0])
        back = proposed_cont.death_log_eval(front, adj_probs[1][2])
        #print(f'moves: {fwd[0] + fwd[1]}, {back[1] + back[0]}')
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    elif front.last_move == 'death':
        fwd = front.death_log_eval(proposed_cont, adj_probs[0][2])
        back = proposed_cont.birth_log_eval(front, adj_probs[1][0])
        #print(f'moves: {fwd[0] + fwd[1]}, {back[1] + back[0]}')
        acc_ratio = (fwd[0] - fwd[1]) - (back[0] - back[1])
    # elif front.last_move == 'stick':
    # acc_ratio = proposed_cont.eval()-front.eval() + cont_move
    else:
        acc_ratio = front.eval()-particle.eval()

    #print(f'acceptance rate is {acc_ratio}')
    #print(f'acceptance ratio: {acc_ratio}')
    return front, weight + min(0,acc_ratio)


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
    max_k = max((p.Gaussian_Mix_Model.k for p in particles), default=-1)
    inddict = {k: [] for k in range(max_k + 2)}  # Preallocate dictionary with empty lists

    for i, p in enumerate(particles):
        inddict[p.Gaussian_Mix_Model.k].append(i)

    absent = sum(1 for v in inddict.values() if not v)  # Count empty lists

    return inddict, absent


def get_probdict(particles, weights, neff):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        essprob = 1-(neff/len(particles))
        print(f'neff = {neff/len(particles)}')
        weights = normalise(weights)
        #sys.stdout.flush()
        #neff = ess(weights, exec=Executor_MPI())
        #sys.stdout.flush()
        #print(f'ess = {neff}')
        #comm.barrier()
        #sys.stdout.flush()

        partsort = get_k_indices(particles)
        inddict = partsort[0]
        print(f'dictionary keys are {inddict.keys()}')
        absent = partsort[1]
        print(f'absent = {absent}')
        if absent != 0:
            std = essprob/absent
        else:
            std = 0

        allprobs = []
        for i in inddict:
            if i == 0:
                allprobs.append(0)
            elif inddict[i] == []:
                allprobs.append(std)
            else :
                allprobs.append((neff/len(particles))*sum([np.exp(weights[j]) for j in inddict[i]]))

        allprobs = allprobs/sum(allprobs)
        print(f'PMF is  {allprobs}')
        probdicts = {0:[0,0,0]}
        for i in range(1, max(inddict.keys())):
            ps = np.array([max(10**-235, allprobs[i - 1]), max(10**-235, allprobs[i]), max(10**-235, allprobs[i + 1])])
            probdicts[i] = ps / sum(ps)



        return probdicts

    else:
        return None


def RJMCMC(init, T, burn):
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

    return path[burn:], k[burn:], bic[burn:]

def nRJMCMC(init, T, burn):
    k = []
    bic = []
    path = [init.get_initial_dist()]
    p = random.uniform(0,1)

    t = 0
    a = 0
    lastnu = None
    while t < T:
        if lastnu is None:
            prop = nRJMCMC_Step(path[-1])
        else:
            prop = nRJMCMC_Step(path[-1], lastnu)
        path.append(prop[0])
        k.append(prop[0].Gaussian_Mix_Model.k)
        bic.append(prop[0].bic())
        print(f'Completed step {t}')
        t += 1
        a += prop[1]
        if prop[2] != lastnu:
            print('Reversal!')
        lastnu = prop[2]

        print(f'Acceptance rate = {np.round(a / t, 2)}')
        print(f'Current components = {path[-1].Gaussian_Mix_Model.k}')
        print(f'Current BIC = {path[-1].bic()}')

    return path[burn:], k[burn:], bic[burn:]


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
        res1, res2 = SMCstep(x, y, adj_probs =[probdict[x.Gaussian_Mix_Model.k-1], probdict[x.Gaussian_Mix_Model.k], [x.Gaussian_Mix_Model.k+1]])
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
        k_path = [current_particles[0].Gaussian_Mix_Model.k]
        bic_path = [current_particles[0].bic()]

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

        if math.log(neff) < math.log(N) + math.log(0.25): #math.log(N) - math.log(2):

            current_particles = pad(current_particles, exec = Executor_MPI())
            current_particles, logWeights = systematic_resampling(
                current_particles, logWeights, mvrs_rng, exec = Executor_MPI())
            comm.barrier()


            current_particles = restore(current_particles)

        if rank == 0:
            k_path.append(get_wt_average_k(current_particles, logWeights))
            bic_path.append(get_wt_av_bic(current_particles, logWeights))

        sys.stdout.flush()

        current_particles, logWeights = one_step_sample_MPI(current_particles, logWeights)
        #print(f'rank = {rank}, after sample weights = {logWeights}')

        all_particles = np.array(comm.gather(current_particles, root = 0)).ravel()
        all_wts = np.array(comm.gather(logWeights, root = 0)).ravel()

        if rank == 0:
            particle_path.append(np.array(all_particles))
            logwt_path.append(np.array(all_wts))

        t+=1


    return particle_path, logwt_path, ess_path, k_path, bic_path


def wt_informed_RJSMC_MPI(init, N, T):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    loc_n = int(N)
    # rank = self.exec.rank

    seed = 0
    mvrs_rng = RNG(seed)
    # rngs = [RNG(i + rank * loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle

    current_particles = [init.get_initial_dist()] * int(N/size)

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
        k_path = [current_particles[0].Gaussian_Mix_Model.k]
        bic_path = [current_particles[0].bic()]

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

        if math.log(neff) < math.log(N) + math.log(0.25):
            current_particles = pad(current_particles, exec=Executor_MPI())
            current_particles, logWeights = systematic_resampling(
                current_particles, logWeights, mvrs_rng, exec=Executor_MPI())
            comm.barrier()

            current_particles = restore(current_particles)

        sys.stdout.flush()

        # Propose new samples from the particle front in a single step
        if rank == 0:
            k_path.append(get_wt_average_k(current_particles, logWeights))
            bic_path.append(get_wt_av_bic(current_particles, logWeights))

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
        return particle_path, logwt_path, ess_path, k_path, bic_path
    else:
        return None, None, None


