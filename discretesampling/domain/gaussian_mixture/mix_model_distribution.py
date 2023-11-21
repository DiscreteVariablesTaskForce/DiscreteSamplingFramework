import numpy as np
from discretesampling.base.types import DiscreteVariableProposal
from discretesampling.domain.gaussian_mixture import util as gmm_util
from scipy.special import beta
from scipy.special import gamma
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import poisson
from scipy.stats import dirichlet
from scipy.stats import invgamma


class UnivariateGMMProposal(DiscreteVariableProposal):

    def __init__(self, means, covs, compwts, g, alpha, la, delta, allocation_structure):
        self.compwts = list(gmm_util.normalise(compwts))
        self.means = means
        self.covs = covs
        self.n_comps = len(self.compwts)
        self.indices = [i for i in range(self.n_comps)]

        'Merge with allocation structure'
        self.data = allocation_structure.data
        self.current_allocations = allocation_structure.proposed_allocations
        self.current_logprob = allocation_structure.proposed_logprob
        self.proposal_allocations = allocation_structure.proposed_allocations
        self.proposal_logprob = allocation_structure.proposed_logprob
        
        'Retrieve or generate hyperparameters for distribution'
        self.g = g
        self.alpha = alpha
        self.la = la
        self.delta = delta

        'Blank data for retrieving move information'
        self.last_action = None
        self.last_split = None
        self.last_merge = None
        self.last_birth = None
        self.last_death = None


    def jump_proposal(self):
        u = np.random.uniform(0,1)
        if u < 0.5:
            return 1
        else:
            return -1

    def dv_sample(self, gmm, alloc, move_type):
        r = self.jump_proposal()

        if move_type == 'split_merge':
            if r == 1:
                print('Splitting!')
                new_gmm = gmm.split(alloc)
            elif r == -1:
                print('Merging!')
                new_gmm = gmm.merge(alloc)
            else:
                print('No discrete move!')
                new_gmm = gmm

        elif move_type == 'birth_death':
            if r == 1:
                print('Birth!')
                new_gmm = gmm.birth(alloc)
            elif r == -1:
                print('Death!')
                new_gmm = gmm.death(alloc)
            else:
                new_gmm = gmm

        elif move_type == 'weights':

            new_gmm = gmm.update_weights(alloc)

        elif move_type == 'allocations':

            alloc.propose_allocation_update(gmm.indices, gmm)

        elif move_type == 'parameters':

            new_gmm = gmm.update_parameters(alloc)

        elif move_type == 'beta_update':

            new_gmm = gmm.update_beta(alloc)

        else:
            print('Discrete move not known, returning original proposal')
            new_gmm = gmm

        return new_gmm.getProposalType()

    def eval(self):
        if self.last_action == 'split':

            pass

        elif self.last_action == 'merge':

            pass

        elif self.last_action == 'birth':

            pass

        elif self.last_action == 'death':

            pass

        else:

            return 0





    def eval_at_x(self, x) -> float:
        return sum([self.compwts[i] * self.eval_component(i, x) for i in range(self.n_modes)])

    def eval_component(self, ind, x) -> float:
        # logprob of a single component of the mixture

        return (((x - self.means[ind]) ** 2) / (2 * self.covs[ind] ** 2)) - np.log(
                np.sqrt(2 * np.pi) * self.covs[ind])

    def eval_likelihood_ratio(self, current, proposed, allocation_structure):
        if proposed.last_action == 'parameters_rejected':
            return 0
        else:
            curr_loglikelihood = 0
            prop_loglikelihood = 0
            for i in range(len(proposed.data)):
                curr_loglikelihood += current.eval_component(allocation_structure.current_data_allocations[i], allocation_structure.data[i])
                prop_loglikelihood += proposed.eval_component(allocation_structure.proposed_data_allocations[i], allocation_structure.data[i])

            return prop_loglikelihood - curr_loglikelihood

    def eval_split_merge(self, current, proposed, type = 'split'):
        c_alloc = current.allocate_data(current.indices)
        p_alloc = proposed.allocate_data(proposed.indices)

        if current.last_action == 'split_rejected':
            return 0
        elif type == 'split' and current.last_split == None:
            print('No split has occured to evaluate!')
            return 0
        elif type == 'merge' and current.last_merge == None:
            print('No merge has occurred to evaluate!')
            return 0
        else:
            if type == 'split':
                flip = 1
                action_index = current.last_split[0]
                uvals = current.last_split[2]
                log_palloc = self.proposal_logprob
                muj = current.means[action_index]
                varj = current.covs[action_index]
                wtj = current.compwts[action_index]
                mu12 = [proposed.means[action_index], proposed.means[action_index+1]]
                var12 = [proposed.covs[action_index], proposed.covs[action_index + 1]]
                wt12 = [proposed.compwts[action_index], proposed.compwts[action_index + 1]]
                alloc_1 = p_alloc[1][action_index]
                alloc_2 = p_alloc[1][action_index + 1]
            else:
                flip=-1
                action_index = current.last_merge[0]
                uvals = current.last_merge[2]
                log_palloc = 0
                mu12 = current.last_merge[1][0]
                var12 = current.last_merge[1][1]
                wt12 = current.last_merge[1][2]
                muj = proposed.means[action_index]
                varj = proposed.covs[action_index]
                wtj = proposed.compwts[action_index]
                alloc_1 = c_alloc[1][action_index]
                alloc_2 = c_alloc[1][action_index+1]

            wtrat_up = (proposed.delta - 1 + alloc_1)*np.log(wt12[0]) + (proposed.delta - 1 + alloc_2)*np.log(wt12[1])
            wtrat_down = (proposed.delta - 1 + alloc_2+alloc_2)*np.log(wtj) + beta(current.delta, current.delta*len(current.compwts))
            wtrat = wtrat_up - wtrat_down

            meanprobs_exp = -(proposed.kappa/2)*((mu12[0]-proposed.zeta)**2+(mu12[1] - proposed.zeta)**2-(muj-proposed.zeta)**2)
            meanprobs = 0.5*(np.log(proposed.kappa) - np.log(2*np.pi)) +  meanprobs_exp

            covs_exp = -proposed.beta*(1/var12[0] + 1/var12[1] - 1/varj)
            covs_fac = -(proposed.alpha+1)*np.log(var12[0]*var12[1]) - np.log(varj)
            covs_fac2 = proposed.alpha*np.log(proposed.beta) - np.log(gamma(proposed.alpha))
            covs =  covs_exp+covs_fac+covs_fac2

            q_u = beta_dist(2,2).pdf(uvals[0])*beta_dist(2,2).pdf(uvals[1])*beta_dist(1,1).pdf(uvals[2])
            jump_ratio = -q_u*log_palloc

            jacob_up = np.log(wtj*var12[0]*var12[1]*np.abs(mu12[0] - mu12[1]))
            jacob_down = np.log(uvals[1]*(1-uvals[1]**2)*uvals[2]*(1-uvals[2])*varj)
            jacob = jacob_up - jacob_down

            comp = np.log(poisson.pmf(len(proposed.compwts), proposed.la)) - np.log(poisson.pmf(len(current.compwts),current.la))
            logprob =  len(proposed.compwts) + comp + wtrat + meanprobs+ covs+ jump_ratio+jacob+current.eval_likelihood_ratio(current,proposed)

            return flip*logprob
    def eval_birth_death(self, current, proposed, type = 'birth'):

        if type == 'birth' and proposed.last_birth == None:
            raise Exception('No birth move to evaluate!')
            return 0
        elif type == 'death' and proposed.last_death == None:
            raise Exception('No death move to evaluate!')
            return 0
        else:
            if type == 'birth':
                wj = self.compwts[proposed.last_birth]
                flip = 1
            else:
                wj = proposed.last_death[2]
                flip = -1

            k0 = len([i for i in range(len(proposed.compwts)) if i not in proposed.data_allocations])
            n = len(proposed.data_allocations)
            delt = proposed.delta -1
            b = beta(current.n_comps * current.delta, current.delta)

            jump = np.log(poisson.pmf(current.n_comps+1, current.la)) - np.log((poisson.pmf(current.n_comps,current.la)*b))
            #print('Jump prob = {}'.format(jump))
            wtprob = delt*np.log(wj) + ((current.n_comps*delt)+n)*np.log(1-wj)
            #print('Delta = {}, weight = {}, logweight = {}'.format(delt, wj, np.log(wj)))
            #print('Weight prob = {}'.format(wtprob))
            propjac = current.n_comps*np.log(1-wj) - (k0+1+beta_dist.pdf(wj,1,current.n_comps))
            #print('Jacobian = {}'.format(propjac))

            log_accprob = flip*(jump+wtprob+propjac)

            return log_accprob


    def eval_weight_allocation(self, current, proposed):

        return proposed.log_palloc - current.log_palloc

    def eval_weight_probs(self, current, proposed):

        return dirichlet.logpdf(proposed.compwts, proposed.dirichlet_parameters) - dirichlet.logpdf(current.compwts, current.dirichlet_parameters)

    def eval_move(self, current, proposed) -> float:

        if proposed.last_action in ['split, merge']:
            self.curr_eval = self.eval_split_merge(current,proposed,type=proposed.last_action)
            print('Evaluated {} move at {}'.format(self.last_action, self.curr_eval))
        elif proposed.last_action in ['birth,death']:
            self.curr_eval = self.eval_birth_death(current, proposed, type= proposed.last_action)
            print('Evaluated {} move at {}'.format(self.last_action, self.curr_eval))
        elif proposed.last_action == 'weights':
            self.curr_eval = self.eval_weight_probs(current,proposed)
            print('Evaluated {} move at {}'.format(self.last_action, self.curr_eval))
        elif proposed.last_action == 'allocations':
            self.curr_eval =  self.eval_weight_allocation(current,proposed)
            print('Evaluated {} move at {}'.format(self.last_action, self.curr_eval))
        elif proposed.last_action == 'parameters':
            self.curr_eval = self.eval_likelihood_ratio(current,proposed)
            print('Evaluated {} move at {}'.format(self.last_action, self.curr_eval))
        else:
            raise Exception('Unknown move type!')

    def eval(self):

        return self.curr_eval()

    def heuristic(self, x, y):

        pass

    def norm(self, x):

        pass

    def sample(self,size = 1):
        samp = []
        wt_cdf = np.cumsum(self.compwts)
        q = np.random.uniform(0,1)
        ind = gmm_util.find_rand(wt_cdf,q)
        samp.append(np.random.normal(self.means, np.sqrt(self.covs)))
        if size == 1:
            return samp[0]
        else:
            return samp

