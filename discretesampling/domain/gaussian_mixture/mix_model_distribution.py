import numpy as np
import math
import copy
from discretesampling.base.types import DiscreteVariableProposal
from discretesampling.base.types import JumpProposal
from discretesampling.domain.gaussian_mixture import util as gmm_util
from scipy.special import beta
from scipy.special import gamma
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import poisson
from scipy.stats import dirichlet
from scipy.stats import norm
from scipy.stats import invgamma


class Split_K(JumpProposal):
    def __init__(self, move='split'):
        self.move = move

    def jump_prob(self,k):
        r = np.random.uniform(0, 1)
        if r < 1/k:
            return 1
        else:
            return -1

    def jump_eval(self, k, movetype):
        if movetype in ['split', 'birth']:
            return 1/k
        else:
            return(1-(1/k))

class RandomFlip(JumpProposal):
    def __init__(self, move ='split'):
        self.move= move


    def jump_prob(self):
        r = np.random.uniform(0,1)
        if r < 0.5:
            return 1
        else:
            return -1

    def jump_eval(self, x, movetype):
        if movetype == 'None':
            return 0
        else:
            return 0.5

class UnivariateGMMProposal(DiscreteVariableProposal):
    def __init__(self, gmm, allocation_structure):
        self.gmm = gmm
        self.alloc = allocation_structure


    def dv_sample(self, current, move_type, jump= RandomFlip()):
        r = jump.jump_prob()

        if move_type == 'split_merge':
            if r == 1:
                proposal = current.gmm.split(current.alloc)
            elif r == -1:
                proposal = current.gmm.merge(current.alloc)
            else:
                print('No discrete move!')
                proposal = [current.gmm, current.alloc]

        elif move_type == 'birth_death':
            if r == 1:
                proposal = current.gmm.birth(current.alloc)
            elif r == -1:
                proposal= current.gmm.death(current.alloc)
            else:
                proposal = [current.gmm, current.alloc]

        elif move_type == 'weights':

            proposal = self.gmm.update_weights(self.alloc)

        elif move_type == 'allocations':

            proposal = self.gmm.propose_allocation_update(self.alloc)

        elif move_type == 'parameters':

            proposal = self.gmm.update_parameters(self.alloc)

        elif move_type == 'beta_update':

            proposal = self.gmm.update_beta(self.alloc)

        else:
            print('Discrete move not known, returning original proposal')
            proposal = [self.gmm, self.alloc]

        return proposal[0].getProposalType(proposal[1])

    def sample(self):

        wt_prop = self.dv_sample(self, 'weights')
        param_prop = wt_prop.dv_sample(wt_prop, 'parameters')
        all_prop = param_prop.dv_sample(param_prop, 'allocations')
        beta_prop = all_prop.dv_sample(all_prop, 'beta_update')

        u = np.random.uniform(0,1)

        if u < 0.5:
            prop_dist = beta_prop.dv_sample(beta_prop, 'split_merge')
        else:
            prop_dist = beta_prop.dv_sample(beta_prop, 'birth_death')

        return prop_dist

    def eval_at_x(self, x) -> float:
        return sum([self.gmm.compwts[i] * norm(self.gmm.means[i], np.sqrt(self.gmm.covs[i])).pdf(x) for i in range(len(self.gmm.covs))])

    def eval_component(self, ind, x) -> float:
        # logprob of a single component of the mixture

        return norm(self.gmm.means[ind], np.sqrt(self.gmm.covs[ind])).pdf(x)

    def eval_likelihood(self):
        likelihood = 0
        for i in self.gmm.indices:
            likelihood += sum([self.gmm.compwts[i]*self.eval_component(i,j) for j in self.alloc.allocation[i]])

        return likelihood
    def eval_likelihood_ratio(self, current, proposed):

            return proposed.eval_likelihood() - current.eval_likelihood()

    def eval_weight_probs(self, current, proposed):

        return dirichlet.logpdf(proposed.compwts, proposed.dirichlet_parameters) - dirichlet.logpdf(current.compwts, current.dirichlet_parameters)

    def eval(self, current, proposed, jump = RandomFlip()):

        current = self
        lr =self.eval_likelihood_ratio(current,proposed)

        if proposed.gmm.last_move in ['split', 'merge']:

            if proposed.gmm.last_move == 'split':
                flip = 1
                ind = proposed.gmm.last_split[0]
                #print('Computing split at index {} of {}'.format(ind, proposed.alloc.allocation.keys()))
                mus = [proposed.gmm.last_split[1][0],proposed.gmm.means[ind],proposed.gmm.means[ind+1]]
                covs = [proposed.gmm.last_split[1][1], proposed.gmm.covs[ind], proposed.gmm.covs[ind + 1]]
                wts = [proposed.gmm.last_split[1][2],proposed.gmm.compwts[ind],proposed.gmm.compwts[ind+1]]
                uvals = proposed.gmm.last_split[2]
                k = len(current.gmm.compwts)
                n1 = len(proposed.alloc.allocation[ind])
                n2 = len(proposed.alloc.allocation[ind + 1])
            elif proposed.gmm.last_move == 'merge':
                flip = -1
                ind = proposed.gmm.last_merge[0]
                #print('Computing merge at indices {}, {} of {}'.format(ind, ind+1, current.alloc.allocation.keys()))
                mus = proposed.gmm.last_merge[1][0] + [proposed.gmm.means[ind]]
                covs = proposed.gmm.last_merge[1][1] + [proposed.gmm.covs[ind]]
                wts = proposed.gmm.last_merge[1][2] + [proposed.gmm.compwts[ind]]
                uvals = proposed.gmm.last_merge[2]
                n1 = len(current.alloc.allocation[ind])
                n2 = len(current.alloc.allocation[ind + 1])
                k = len(proposed.gmm.compwts)

            musdiff = (mus[1] - proposed.alloc.zeta)**2 + (mus[2] - proposed.alloc.zeta)**2 - (mus[0]-proposed.alloc.zeta)**2

            'Compute p(k)'
            p_k = math.log(poisson.pmf(k+1, proposed.gmm.la)) +math.log(k+1) - math.log(poisson.pmf(k,proposed.gmm.la))

            'Compute p(w;k)'
            w1 = n1 + self.gmm.delta - 1
            w2 = n2 + self.gmm.delta - 1
            w0 = w1+ w2
            p_wk = w1*wts[1] + w2*wts[2] - w0*wts[0] - math.log(beta(self.gmm.delta, self.gmm.delta*len(self.gmm.compwts)))
            #print('Weight probs: {}'.format(p_wk))

            'compute p(mu;w,k)'
            p_muwk = (0.5*math.log(self.alloc.kappa)) - math.log(2*np.pi) + (0.5*self.alloc.kappa)*musdiff
            #print('Mean probs: {}'.format(p_muwk))

            'compute p(s;w,k)'
            p_swk_norm = self.gmm.alpha*math.log(self.gmm.beta) - math.log(gamma(self.gmm.alpha))
            p_swk_fact = (2*self.gmm.alpha-2)*(math.log(covs[0]) +math.log(covs[1]))
            p_swk = p_swk_norm + p_swk_fact - self.gmm.beta*(covs[1]**-1 + covs[2]**-1 - covs[0]**-1)
            #print('Cov probs: {}'.format(p_swk))

            'compute move probability'
            q = beta_dist(2,2).pdf(uvals[0])*beta_dist(2,2).pdf(uvals[1])*beta_dist(1,1).pdf(uvals[2])
            p_jump = math.log(jump.jump_eval(current, 'Forward')) - math.log(jump.jump_eval(current, 'Reverse'))
            p_move = p_jump + q
            #print('Move prob: {}'.format(p_move))

            'compute Jacobian'
            if proposed.gmm.last_move == 'split':
                jac_up = math.log(wts[0]) + math.log(np.abs(mus[1]-mus[2]))
                jac_down = math.log(uvals[1]) + math.log(1-uvals[1]**2) + math.log(uvals[2]) + math.log(1-uvals[2])
                J = jac_up - jac_down + math.log(covs[1]) +math.log(covs[2]) - math.log(covs[0])
            else:
                J = 1

            logps = J + lr + flip*(p_k + p_wk + p_muwk + p_swk - p_move - proposed.alloc.log_palloc)
            return logps

        elif proposed.gmm.last_move in ['birth', 'death']:

            if proposed.gmm.last_move == 'birth':
                flip = 1
                forward = proposed
                back = current
                ind = proposed.gmm.last_birth
                #print('Computing birth at index {}'.format(ind))
                w0 = proposed.gmm.compwts[ind]
                k = proposed.gmm.n_comps
                k0 = len(current.alloc.get_empties())
            elif proposed.gmm.last_move == 'death':
                flip = -1
                forward = current
                back = proposed
                ind = proposed.gmm.last_death
                w0 = current.gmm.compwts[ind]
                k = current.gmm.n_comps
                k0 = len(proposed.alloc.get_empties())

            'Compute p(k)'
            p_k = math.log(poisson.pmf(k+1, proposed.gmm.la)) +math.log(k+1) - math.log(poisson.pmf(k,proposed.gmm.la))

            'Compute beta factor'
            bet = (-math.log(beta(self.gmm.delta, k*self.gmm.delta))) + (self.gmm.delta*math.log(w0)) + self.gmm.delta*(k+1) + (len(self.alloc.data)*(1-w0))

            'Compute jump ratio and jacobian'
            bk = jump.jump_eval(forward, 'Forward')
            dk1 = jump.jump_eval(forward, 'Reverse')

            j = (dk1*len(back.gmm.means)*(1-w0))/(bk*(k0 + 1)*len(back.gmm.means)*np.random.beta(1, len(back.gmm.means)))

            return flip*(p_k + j - bet)

        else:
            return 0

    def heuristic(self, x, y):

        pass

    def norm(self, x):

        pass



