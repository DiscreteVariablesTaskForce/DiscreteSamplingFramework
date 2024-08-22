import numpy as np
import math
import copy
from scipy.stats import norm
from numpy import random
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import dirichlet
from scipy.stats import poisson
from scipy.stats import invgamma
from scipy.special import logsumexp

import sys
sys.path.append('C:/Users/mattb242/Desktop/Projects/reversible_jump/local_code/DiscreteSamplingFramework')


from discretesampling.domain.gaussian_mixture import util
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model

class Data_Allocation:
    def __init__(self, allocation):

        self.allocation = allocation
        self.k = max(allocation.keys())

    def all_data(self):
        dat = []
        for i in self.allocation:
            dat.extend(self.allocation[i])

        return dat

    def component_means(self):
        mean_dict = {}
        for i in self.allocation.keys():
            mean_dict[i] = np.average(self.allocation[i])

        return mean_dict

    def component_variances(self):
        var_dict = {}
        for i in self.allocation.keys():
            var_dict[i] = np.var(self.allocation[i])

        return var_dict

    def get_empties(self):

        return [i for i in self.allocation.keys() if self.allocation[i] == []]

    def insert_component(self, n):
        """
        Inserts an empty component at index key+1, shifting all other keys to the right
        """
        new_dict = {}
        if n == 0:
            new_dict[0] = []
            for key, value in self.allocation.items():
                new_dict[key + 1] = value

        elif n == max(self.allocation.keys()) + 1:
            for key, value in self.allocation.items():
                new_dict[key] = value

            new_dict[n] = []

        else:
            for key, value in self.allocation.items():
                if key < n:
                    new_dict[key] = value
                elif key >= n:
                    new_dict[key + 1] = value

            new_dict[n] = []


        return new_dict

    def merge_components(self, n):
        if n > max(self.allocation.keys())-1:
            raise Exception(f'At least one of your merge indices does not exist at {n}')

        merged_list = self.allocation[n] + self.allocation[n+1]

        new_dict = {}

        for key, value in self.allocation.items():
            if key < n:
                new_dict[key] = value
            elif key == n:
                new_dict[key] = merged_list
            elif key > n+1:
                new_dict[key-1] = self.allocation[key]

        return Data_Allocation(new_dict)

class GMM_Distribution():
    def __init__(self, Gaussian_Mix_Model, Data_Allocation, la, delta, alpha, g, ep_h, ep_k):

        self.Data_Allocation = Data_Allocation
        self.Gaussian_Mix_Model = Gaussian_Mix_Model

        #Fixed hyperparameters
        self.la = la
        self.delta = delta
        self.alpha = alpha
        self.g = g
        self.ep_h = ep_h
        self.ep_k = ep_k

        #Derived hyperparameters
        self.delts = [self.delta]*self.Gaussian_Mix_Model.k
        self.zeta = np.median(self.Data_Allocation.all_data())
        self.R = np.ptp(self.Data_Allocation.all_data())
        self.h = ep_h*self.R**-2
        self.kappa = ep_k*self.R**-2

        sumvar = sum([i[1] ** -1 for i in self.Gaussian_Mix_Model.components])
        self.beta = gamma.rvs((self.alpha * self.Gaussian_Mix_Model.k) + self.g, scale = (self.h + sumvar))

        self.last_move = None
        self.aux_rand = None

    def order_components(self):

        new_comps = copy.copy(self.Gaussian_Mix_Model.components)
        new_comps = sorted(new_comps, key=lambda x:x[0])

        new_allocs = {}
        for i in range(len(new_comps)):
            new_allocs[i] = self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(new_comps[i])]

        self.Gaussian_Mix_Model = Gaussian_Mix_Model(new_comps)
        self.Data_Allocation = Data_Allocation(new_allocs)

        return self

    def compute_logprob(self,data):
        lp = 0
        for i in data:
            lp += sum([j[2]*norm.logpdf(i, j[0], j[1]) for j in self.Gaussian_Mix_Model.components])

        return lp

    def merge(self):
        if self.Gaussian_Mix_Model.k == 1:
            new_dist = copy.deepcopy(self)
            new_dist.last_move = 'merge_rejected'

        else:
            merge_ind = random.choice([i for i in range(self.Gaussian_Mix_Model.k-1)])

            mwts =  [self.Gaussian_Mix_Model.components[merge_ind][2], self.Gaussian_Mix_Model.components[merge_ind+1][2]]
            mmu = [self.Gaussian_Mix_Model.components[merge_ind][0], self.Gaussian_Mix_Model.components[merge_ind+1][0]]
            mvar = [self.Gaussian_Mix_Model.components[merge_ind][1], self.Gaussian_Mix_Model.components[merge_ind+1][1]]

            #create merged component
            new_wt = sum(mwts)
            new_mu = (mmu[0]*mwts[0] + mmu[1]*mwts[1])/new_wt
            new_var = (mwts[0]*(mmu[0]**2 + mvar[0]) + mwts[1]*(mmu[1]**2 + mvar[1]) - (new_wt*new_mu**2))/new_wt

            #create new distribution with components merged
            newcomps = []
            for i in range(self.Gaussian_Mix_Model.k - 1):
                if i < merge_ind:
                    newcomps.append(self.Gaussian_Mix_Model.components[i])
                elif i == merge_ind:
                    newcomps.append([new_mu, new_var, new_wt])
                elif i > merge_ind:
                    newcomps.append(self.Gaussian_Mix_Model.components[i+1])

            new_gmm = Gaussian_Mix_Model(newcomps)
            new_alloc = self.Data_Allocation.merge_components(merge_ind)

            new_dist = GMM_Distribution(new_gmm, new_alloc, self.la, self.delta, self.alpha, self.g, self.ep_h, self.ep_k)
            new_dist.last_move = 'merge'
            new_dist.previous_distribution = self

        return new_dist

    def split(self):
        #Choose an index
        split_ind = random.choice([i for i in range(self.Gaussian_Mix_Model.k)])
        split_dat = self.Data_Allocation.allocation[split_ind]


        u_1 = beta.rvs(2,2)
        u_2 = beta.rvs(2,2)
        u_3 = beta.rvs(1,1)

        wt_1 = self.Gaussian_Mix_Model.wts[split_ind]*u_1
        wt_2 = self.Gaussian_Mix_Model.wts[split_ind]*(1-u_1)

        mu_1 = self.Gaussian_Mix_Model.means[split_ind] - (u_2 * np.sqrt(self.Gaussian_Mix_Model.vars[split_ind]*(wt_2/wt_1)))
        mu_2 = self.Gaussian_Mix_Model.means[split_ind] + u_2 * np.sqrt(
                self.Gaussian_Mix_Model.vars[split_ind] * (wt_1 / wt_2))

        if self.Gaussian_Mix_Model.k == 1:
            new_means = [mu_1, mu_2]
        elif split_ind == 0:
            new_means = np.concatenate(([mu_1, mu_2], self.Gaussian_Mix_Model.means[1:]))
        elif split_ind == len(self.Gaussian_Mix_Model.means)-1:
            new_means = np.concatenate((self.Gaussian_Mix_Model.means[:split_ind], [mu_1, mu_2]))
        else:
            new_means = np.concatenate((self.Gaussian_Mix_Model.means[:split_ind], [mu_1, mu_2], self.Gaussian_Mix_Model.means[split_ind+1:]))

        if util.is_ordered(new_means):
            if self.Gaussian_Mix_Model.k == 1:
                new_wts = [wt_1, wt_2]
            elif split_ind == 0:
                new_wts = np.concatenate(([wt_1, wt_2], self.Gaussian_Mix_Model.wts[1:]))
            elif split_ind == len(self.Gaussian_Mix_Model.means) - 1:
                new_wts = np.concatenate((self.Gaussian_Mix_Model.wts[:split_ind], [wt_1, wt_2]))
            else:
                new_wts = np.concatenate((self.Gaussian_Mix_Model.wts[:split_ind], [wt_1, wt_2],
                                                self.Gaussian_Mix_Model.wts[split_ind + 1:]))



            var_1 = u_3*(1-u_2**2)*self.Gaussian_Mix_Model.vars[split_ind]*(self.Gaussian_Mix_Model.wts[split_ind]/wt_1)
            var_2 = (1-u_3) * (1 - u_2 ** 2) * self.Gaussian_Mix_Model.vars[split_ind] * (
                            self.Gaussian_Mix_Model.wts[split_ind] / wt_2)
            if self.Gaussian_Mix_Model.k == 1:
                new_vars = [var_1, var_2]
            elif split_ind == 0:
                new_vars = np.concatenate(([var_1, var_2], self.Gaussian_Mix_Model.vars[1:]))
            elif split_ind == len(self.Gaussian_Mix_Model.vars) - 1:
                new_vars = np.concatenate((self.Gaussian_Mix_Model.vars[:split_ind], [var_1, var_2]))
            else:
                new_vars = np.concatenate((self.Gaussian_Mix_Model.vars[:split_ind], [var_1, var_2],
                                                self.Gaussian_Mix_Model.vars[split_ind + 1:]))

            new_comps = [[new_means[i], new_vars[i], new_wts[i]] for i in range(len(new_means))]

            new_gmm = Gaussian_Mix_Model(new_comps)

            data_alloc = [[], []]
            s = wt_1 + wt_2
            nwts = [wt_1/s, wt_2/s]
            for i in split_dat:
                logprobs = [math.log(nwts[0]) + norm.logpdf(i, mu_1, np.sqrt(var_1)), math.log(nwts[1]) + norm.logpdf(i, mu_2, np.sqrt(var_2))]
                probsum = logsumexp(logprobs)
                normprobs = logprobs - probsum
                assign_index = util.assign_from_pmf(np.exp(normprobs))
                data_alloc[assign_index].append(i)

            new_alloc = {}
            for i in self.Data_Allocation.allocation:
                if i < split_ind:
                    new_alloc[i] = self.Data_Allocation.allocation[i]
                elif i == split_ind:
                    new_alloc[i] = data_alloc[0]
                    new_alloc[i+1] = data_alloc[1]
                elif i > split_ind:
                    new_alloc[i+1] = self.Data_Allocation.allocation[i]

            newalloc = Data_Allocation(new_alloc)

            new_dist = GMM_Distribution(new_gmm, newalloc, self.la, self.delta, self.alpha, self.g, self.ep_h, self.ep_k)
            new_dist.last_move = 'split'
            return new_dist

        else:
            new_dist = copy.deepcopy(self)
            new_dist.last_move = 'split_rejected'
            return new_dist

    def birth(self):

        #Generate random new birth elements
        nmu = norm.rvs(self.zeta, np.sqrt(self.kappa**-1))
        nvar = invgamma.rvs(self.alpha, scale = self.beta)
        nwt = beta.rvs(1, self.Gaussian_Mix_Model.k)

        insertion = self.Gaussian_Mix_Model.insert_new_component([nmu, nvar, nwt])
        s = sum(insertion[0].wts)
        for i in insertion[0].components:
            i[2] = i[2]/s

        new_gmm = insertion[0]

        newalloc = self.Data_Allocation.insert_component(insertion[1])

        new_alloc = Data_Allocation(newalloc)

        new_dist = GMM_Distribution(new_gmm, new_alloc, self.la, self.delta, self.alpha, self.g, self.ep_h, self.ep_k)

        new_dist.last_move = 'birth'
        new_dist.previous_distribution = self

        return new_dist

    def death(self):

        empties = [i for i in self.Data_Allocation.allocation.keys() if self.Data_Allocation.allocation[i] == []]

        if not empties or self.Gaussian_Mix_Model.k == 1:
            new_dist = GMM_Distribution(self.Gaussian_Mix_Model, self.Data_Allocation, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                        self.ep_k)
            new_dist.last_move = 'death_rejected'
        else:
            kill = random.choice(empties)
            newcomps = []
            newalloc = {}
            for i in range(self.Gaussian_Mix_Model.k):
                if i < kill:
                    newcomps.append(self.Gaussian_Mix_Model.components[i])
                    newalloc[i] = self.Data_Allocation.allocation[i]

                elif i > kill:
                    newcomps.append(self.Gaussian_Mix_Model.components[i])
                    newalloc[i-1] = self.Data_Allocation.allocation[i]

            s = sum([i[2] for i in newcomps])
            for i in newcomps:
                i[2] = i[2]/s

            new_gmm = Gaussian_Mix_Model(newcomps)
            new_alloc = Data_Allocation(newalloc)

            new_dist = GMM_Distribution(new_gmm, new_alloc, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                        self.ep_k)

            new_dist.Gaussian_Mix_Model.normalise_weights()
            new_dist.last_move = 'death'
            new_dist.previous_distribution = self

        return new_dist

    def wt_gibbs_update(self):


        new_params = np.array([self.delts[i] + len(self.Data_Allocation.allocation[i]) for i in self.Data_Allocation.allocation])

        new_wts = dirichlet.rvs(new_params)[0]

        newcomps = []
        for i in range(len(new_wts)):
            newcomps.append([self.Gaussian_Mix_Model.components[i][0], self.Gaussian_Mix_Model.components[i][1], new_wts[i]])


        newmodel = Gaussian_Mix_Model(newcomps)
        new_dist = GMM_Distribution(newmodel, self.Data_Allocation, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                        self.ep_k)

        return new_dist

    def mu_gibbs_update(self):

        s_i = [sum(self.Data_Allocation.allocation[i]) for i in self.Data_Allocation.allocation.keys()]
        n_i = [len(self.Data_Allocation.allocation[i]) for i in self.Data_Allocation.allocation.keys()]


        newmeans = [norm.rvs(((s_i[i]/self.Gaussian_Mix_Model.vars[i])+(self.zeta*self.kappa))/((n_i[i]/self.Gaussian_Mix_Model.vars[i])+self.kappa), (np.sqrt((n_i[i]/self.Gaussian_Mix_Model.vars[i])+self.kappa))**-1) for i in range(len(self.Gaussian_Mix_Model.means))]

        if util.is_ordered(newmeans):
            newcomps = [[newmeans[i], self.Gaussian_Mix_Model.vars[i], self.Gaussian_Mix_Model.wts[i]] for i in range(self.Gaussian_Mix_Model.k)]

            newmodel = Gaussian_Mix_Model(newcomps)
            new_dist = GMM_Distribution(newmodel, self.Data_Allocation, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                    self.ep_k)


            return new_dist

        else:
            return self
    def var_gibbs_update(self):

        newcomps = []
        for i in self.Gaussian_Mix_Model.components:
            idat = self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)]
            n_i = len(idat)
            if n_i == 0:
                var_i = 0
            else:
                var_i = sum([(j-i[0])**2 for j in idat])

            new_s_i = invgamma.rvs(self.alpha + (n_i/2), scale = self.beta + (var_i/2))
            newcomps.append([i[0], new_s_i, i[2]])

        newmodel = Gaussian_Mix_Model(newcomps)
        new_dist = GMM_Distribution(newmodel, self.Data_Allocation, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                    self.ep_k)

        return new_dist

    def beta_update(self):

        new_dist = copy.deepcopy(self)
        sumvar = sum([i[1]**-1 for i in self.Gaussian_Mix_Model.components])
        new_dist.beta = gamma.rvs((self.alpha*self.Gaussian_Mix_Model.k)+self.g, scale= (self.h + sumvar))

        return new_dist

    def allocation_update(self):
        newdict = self.Gaussian_Mix_Model.allocate_data(self.Data_Allocation.all_data())

        new_alloc = Data_Allocation(newdict)
        new_dist = GMM_Distribution(self.Gaussian_Mix_Model, new_alloc, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                    self.ep_k)

        return new_dist

    def continuous_forward_sample(self, fixed_beta = False):

        n_1 = self.wt_gibbs_update()
        n_2 = n_1.mu_gibbs_update()
        n_3 = n_2.var_gibbs_update()
        n_4 = n_3.allocation_update()

        if not fixed_beta:
            n_5 = n_4.beta_update()
            #print('Beta updated from {} to {}'.format(n_4.beta, n_5.beta))
            return n_5
        else:
            return n_4

    def continuous_forward_eval(self):

        wtprob = dirichlet.logpdf(self.Gaussian_Mix_Model.wts, np.array([self.delts[i] + len(self.Data_Allocation.allocation[i]) for i in self.Data_Allocation.allocation]))
        muprob = 0

        for i in self.Gaussian_Mix_Model.components:
            s_i = sum(self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)])
            n_i = len(self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)])
            muprob += norm.logpdf(i[0], ((self.zeta*self.kappa)+(i[1]**-1*s_i))/((i[1]**-1*n_i)+self.kappa), np.sqrt(1/(self.kappa + (i[1]**-1*n_i))))


        varprob = 0

        for i in self.Gaussian_Mix_Model.components:
            idat = self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)]
            n_i = len(idat)
            var_i = np.var(idat)*n_i

            varprob += invgamma.logpdf(i[1], self.alpha+(n_i/2), scale= (self.beta + (var_i/2)))

        return wtprob + muprob + varprob

    def discrete_forward_sample(self, move_pmf=[0.5, 0, 0.5], disc_pmf=[0.5, 0.5]):

        move_choice = util.assign_from_pmf(move_pmf)
        move = util.assign_from_pmf(disc_pmf)
        if move_choice == 0:
            if move == 0:
                return self.merge()
            else:
                return self.death()
        elif move_choice == 1:
            return self
        else:
            if move == 0:
                return self.split()
            else:
                return self.birth()

    def compute_parameter_priors(self):

        kprob = poisson.logpmf(self.Gaussian_Mix_Model.k, self.la)
        log_mueval = sum([norm.logpdf(i, self.zeta, np.sqrt(self.kappa ** -1)) for i in np.around(self.Gaussian_Mix_Model.means, 4)])
        log_vareval = sum([invgamma.logpdf(i, self.alpha, scale= self.beta) for i in np.around(self.Gaussian_Mix_Model.vars, 8)])
        dparams = [self.delta + len(self.Data_Allocation.allocation[i]) for i in self.Data_Allocation.allocation]
        s = sum(self.Gaussian_Mix_Model.wts)
        normwts = []
        for i in self.Gaussian_Mix_Model.wts:
            normwts.append(i/s)

        log_wteval = dirichlet.logpdf(normwts, dparams)

        return kprob, log_mueval, log_vareval, log_wteval


    def split_log_eval(self, previous, split_prob):
        """

        Returns: Tuple whose first entry is the probability of all continuous parameters and whose second is the jump probability
        """


        #check if current distribution could have been derived from previous distribution by a split at all
        if self.Gaussian_Mix_Model.k - previous.Gaussian_Mix_Model.k != 1:
            raise Exception('Split Eval Error: More than one extra component in current distribution')
        elif self.Gaussian_Mix_Model.k > 2 and util.matchlist(self.Gaussian_Mix_Model.means, previous.Gaussian_Mix_Model.means) != 2:
            raise Exception('Split Eval Error: current distribution differs from previous at more than two parameter vectors')
        #if previous distribution is compatible, find the  split index
        else:
            ind = 0
            for i in range(self.Gaussian_Mix_Model.k):
                if self.Gaussian_Mix_Model.components[i] == previous.Gaussian_Mix_Model.components[i]:
                    ind+=1
                else:
                    break
            #if ind == self.Gaussian_Mix_Model.k:
                #ind+=1

            #isolate the split component data from current and previous
            splits = [self.Gaussian_Mix_Model.components[ind], self.Gaussian_Mix_Model.components[ind + 1]]

            sc = previous.Gaussian_Mix_Model.components[ind]

            #LIKELIHOOD AND PRIOR COMPUTATIONS
            #---------------------------------
            #Compute the likelihood logp(y | k, w, z, mu, sigma)
            ev = self.eval()

            #compute current priors p(k), p(mu), p(sigma) and p(w)
            priors = sum([i for i in self.compute_parameter_priors()])

            #JUMP PROBABILITY COMPUTATIONS BELOW#
            # compute selected auxiliary variables and thus q(u)
            u_1 = splits[0][2] / sc[2]
            u_2 = ((splits[1][0] - sc[0]) / (np.sqrt(sc[1]))) * np.sqrt(splits[1][2] / splits[0][2])
            u_3 = (splits[0][1]*splits[0][2])/((1-u_2**2)*sc[1]*sc[2])
            us = [u_1, u_2, u_3]
            qprob = beta.logpdf(u_1, 2, 2) + beta.logpdf(u_2, 2, 2) + beta.logpdf(u_3, 1, 1)

            #compute the allocation probability of the split
            palloc = 0
            normed_split_weights = util.normalise([splits[0][2], splits[1][2]])
            for i in self.Data_Allocation.allocation[ind]:
                palloc += math.log(normed_split_weights[0]) + norm.logpdf(i, splits[0][0], np.sqrt(splits[0][1]))
            for i in self.Data_Allocation.allocation[ind + 1]:
                palloc += math.log(normed_split_weights[1]) + norm.logpdf(i, splits[1][0], np.sqrt(splits[1][1]))

            #compute jacobian of split function
            J = (sc[2]*np.abs(splits[1][0] - splits[0][0])*splits[1][1] +splits[0][1]) /((
                sc[1]*(1 - (us[1] ** 2))*us[2] *(1 - us[2])))

            p_xy = ev + priors

            r_x = qprob + palloc + math.log(split_prob) - math.log(J)

            return p_xy, r_x

    def merge_log_eval(self, previous, mergeprob):

        # check if current distribution could have been derived from previous distribution by a split at all
        if previous.Gaussian_Mix_Model.k - self.Gaussian_Mix_Model.k != 1:
            raise Exception('Merge Eval Error: More than one extra component in previous distribution')
        elif previous.Gaussian_Mix_Model.k > 2 and util.matchlist(previous.Gaussian_Mix_Model.means,
                                           self.Gaussian_Mix_Model.means) != 2:
            raise Exception('Merge Eval Error: Distributions differ at more than two parameter vectors')
        # if previous distribution is compatible, find the  merge index
        else:
            ind = 0
            for i in range(previous.Gaussian_Mix_Model.k):
                if previous.Gaussian_Mix_Model.components[i] == self.Gaussian_Mix_Model.components[i]:
                    ind += 1
                else:
                    break

            # isolate the split components from the previous distribution that were merged
            splits = [previous.Gaussian_Mix_Model.components[ind], previous.Gaussian_Mix_Model.components[ind + 1]]
            sc = self.Gaussian_Mix_Model.components[ind]

            #LIKELIHOOD AND PRIOR COMPUTATIONS
            #_________________________________
            # Compute the likelihood logp(y | k, w, z, mu, sigma)
            ev = self.eval()

            # compute current priors p(k), p(mu), p(sigma) and p(w)
            priors = sum([i for i in self.compute_parameter_priors()])

            p_xy = ev + priors

            #Jump is deterministic, so probability is just raw jump probability for merge
            r_x = math.log(mergeprob)


            return p_xy, r_x

    def birth_log_eval(self, previous, birthprob):

        if self.Gaussian_Mix_Model.k - previous.Gaussian_Mix_Model.k != 1:
            raise Exception('Birth eval error: More than one extra component in current distribution')
        elif util.matchlist(self.Gaussian_Mix_Model.means, previous.Gaussian_Mix_Model.means) != 1:
            raise Exception('Birth eval error: distributions differ at more than one parameter vectors')
            # if previous distribution is compatible, find the  birth index
        else:
            ind = 0
            for i in range(previous.Gaussian_Mix_Model.k):
                if previous.Gaussian_Mix_Model.components[i] == self.Gaussian_Mix_Model.components[i]:
                    ind += 1
                else:
                    break

            sc = self.Gaussian_Mix_Model.components[ind]
            ev = self.eval()

            # compute current priors p(k), p(mu), p(sigma) and p(w)
            priors = sum([i for i in self.compute_parameter_priors()])

            p_xy = ev + priors
            r_x = math.log(birthprob) + beta.logpdf(sc[2], 1, self.Gaussian_Mix_Model.k) - (previous.Gaussian_Mix_Model.k*math.log(1-sc[2]))

        return p_xy, r_x

    def death_log_eval(self, previous, deathprob):

        if previous.Gaussian_Mix_Model.k - self.Gaussian_Mix_Model.k != 1:
            raise Exception('Death Eval Error: Previous distribution has more than one extra component')
        elif util.matchlist(previous.Gaussian_Mix_Model.means, self.Gaussian_Mix_Model.means) != 1:
            raise Exception('Death Eval Error: Distributions differ at more than one parameter vector')
            # if previous distribution is compatible, find the  birth index

        k_0 = len(previous.Data_Allocation.get_empties())

        ev = self.eval()

        # compute current priors p(k), p(mu), p(sigma) and p(w)
        priors = sum([i for i in self.compute_parameter_priors()])

        p_xy = ev + priors
        r_x = math.log(deathprob) - math.log(k_0)
        return p_xy, r_x
    def discrete_forward_eval(self, splitprob, mergeprob, birthprob, deathprob):

        if self.last_move == 'split':

            return self.split_log_eval(self.previous_distribution, splitprob)

        elif self.last_move == 'birth':

            return self.birth_log_eval(self.previous_distribution, mergeprob)

        elif self.last_move == 'merge':

            return self.merge_log_eval(self.previous_distribution, birthprob)

        elif self.last_move == 'death':

            return self.death_log_eval(self.previous_distribution, deathprob)

        else:
            return 0, 0


    def eval(self):
        eval = 0
        for i in self.Data_Allocation.allocation:
            eval += sum([norm.logpdf(j, self.Gaussian_Mix_Model.means[i], np.sqrt(self.Gaussian_Mix_Model.vars[i])) for j in self.Data_Allocation.allocation[i]])

        return eval

