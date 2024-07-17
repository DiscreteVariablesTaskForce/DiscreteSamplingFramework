import numpy as np
import math
import copy
from scipy.stats import norm
from numpy import random
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import dirichlet
from scipy.special import beta as betfunc
from scipy.stats import poisson

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
        self.h = ep_h*self.R**2
        self.kappa = ep_k*self.R**2

        self.beta = gamma.rvs(self.g, scale = (1/self.h))

        self.last_move = None
        self.previous_distribution = None

    def order_components(self):

        new_comps = copy.copy(self.Gaussian_Mix_Model.components)
        new_comps = sorted(new_comps, key=lambda x:x[0])

        new_allocs = {}
        for i in range(len(new_comps)):
            new_allocs[i] = self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(new_comps[i])]

        self.Gaussian_Mix_Model = Gaussian_Mix_Model(new_comps)
        self.Data_Allocation = Data_Allocation(new_allocs)
    def compute_logprob(self,data):
        lp = 0
        for i in data:
            lp += sum([j[2]*norm.logpdf(i, j[0], j[1]) for j in self.Gaussian_Mix_Model.components])

        return lp

    def merge(self):
        if self.Gaussian_Mix_Model.k == 1:
            new_dist = GMM_Distribution(self.Gaussian_Mix_Model, self.Data_Allocation, self.la, self.delta, self.alpha,
                                        self.g, self.ep_h,
                                        self.ep_k)
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
            newalloc = {}
            for i in range(self.Gaussian_Mix_Model.k - 1):
                if i < merge_ind:
                    newcomps.append(self.Gaussian_Mix_Model.components[i])
                    newalloc[i] = self.Data_Allocation.allocation[i]
                elif i == merge_ind:
                    newcomps.append([new_mu, new_var, new_wt])
                    newalloc[i] = self.Data_Allocation.allocation[i] + self.Data_Allocation.allocation[i+1]
                else:
                    newcomps.append(self.Gaussian_Mix_Model.components[i+1])
                    newalloc[i] = self.Data_Allocation.allocation[i+1]

            new_gmm = Gaussian_Mix_Model(newcomps)
            new_alloc = Data_Allocation(newalloc)

            new_dist = GMM_Distribution(new_gmm, new_alloc, self.la, self.delta, self.alpha, self.g, self.ep_h, self.ep_k)
            new_dist.last_move = 'merge'
            new_dist.previous_distribution = self

        return new_dist

    def split(self):
        split_ind = random.choice([i for i in range(self.Gaussian_Mix_Model.k)])
        split_dat = self.Data_Allocation.allocation[split_ind]

        print('Old component: {}'.format(self.Gaussian_Mix_Model.components[split_ind]))

        ordered = False
        while not ordered:
            us = [beta.rvs(2,2), beta.rvs(2,2), beta.rvs(1,1)]
            sc = self.Gaussian_Mix_Model.components[split_ind]


            nwt = [sc[2]*us[0], (1-us[0])*sc[2]]
            normwt = np.array(nwt)/sum(nwt)
            nmu = [sc[0] - (np.sqrt(sc[1])*us[1]*(np.sqrt(nwt[1]/nwt[0]))), sc[0] + (np.sqrt(sc[1])*us[1]*(np.sqrt(nwt[0]/nwt[1])))]
            nvar = [us[2]*(1-us[1]**2)*sc[1]*(sc[2]/nwt[0]), (1-us[2])*(1-us[1]**2)*sc[1]*(sc[2]/nwt[1])]

            if nmu[0] < self.Gaussian_Mix_Model.components[split_ind][0] < nmu[1]:
                ordered = True

        comp_1 = [nmu[0], nvar[0], nwt[0]]
        comp_2 = [nmu[1], nvar[1], nwt[1]]
        print('generated u = {}'.format(us))
        print('resulting components {}, {}'.format(comp_1, comp_2))

        tempgmm = Gaussian_Mix_Model([[nmu[0], nvar[0], normwt[0]], [nmu[1], nvar[1], normwt[1]]])
        tempdat = tempgmm.allocate_data(split_dat)

        newcomps = []
        newalloc = {}

        for i in range(self.Gaussian_Mix_Model.k):
            if i < split_ind:
                newcomps.append(self.Gaussian_Mix_Model.components[i])
                newalloc[i] = self.Data_Allocation.allocation[i]
            elif i == split_ind:
                newcomps.extend([comp_1, comp_2])
                newalloc[i] = tempdat[0][0]
                newalloc[i+1] = tempdat[0][1]
            else:
                newcomps.append(self.Gaussian_Mix_Model.components[i])
                newalloc[i+1] = self.Data_Allocation.allocation[i]

        new_gmm = Gaussian_Mix_Model(newcomps)
        new_alloc = Data_Allocation(newalloc)

        new_dist = GMM_Distribution(new_gmm, new_alloc, self.la, self.delta, self.alpha, self.g, self.ep_h, self.ep_k)
        new_dist.last_move = 'split'
        new_dist.previous_distribution = self

        return new_dist

    def birth(self):

        nmu = norm.rvs(self.zeta, self.kappa)
        nvar = 1/gamma.rvs(self.alpha, self.beta)
        nwt = beta.rvs(1, self.Gaussian_Mix_Model.k)

        print('New component is {}'.format([nmu, nvar, nwt]))
        print('Old components are {}'.format(self.Gaussian_Mix_Model.components))

        new_components = []
        newalloc = {}

        if nmu > max(self.Gaussian_Mix_Model.means):
            new_components = self.Gaussian_Mix_Model.components + [[nmu, nvar, nwt]]
            insert_index = self.Gaussian_Mix_Model.k
            newalloc = copy.deepcopy(self.Data_Allocation.allocation)
            newalloc[max(newalloc.keys())+1]=[]
        else:
            for i in range(self.Gaussian_Mix_Model.k):
                if self.Gaussian_Mix_Model.components[i][0] < nmu:
                    new_components.append(self.Gaussian_Mix_Model.components[i])
                    newalloc[i] = self.Data_Allocation.allocation[i]
                else:
                    new_components.append([nmu, nvar, nwt])
                    newalloc[i] = []
                    insert_index = i
                    break

        if insert_index < self.Gaussian_Mix_Model.k:
            for i in self.Gaussian_Mix_Model.components[insert_index:]:
                new_components.append(i)
                newalloc[self.Gaussian_Mix_Model.components.index(i)+1] = self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)]


        new_gmm = Gaussian_Mix_Model(new_components)
        new_alloc = Data_Allocation(newalloc)

        new_dist = GMM_Distribution(new_gmm, new_alloc, self.la, self.delta, self.alpha, self.g, self.ep_h, self.ep_k)
        new_dist.Gaussian_Mix_Model.normalise_weights()
        new_dist.last_move = 'birth'
        new_dist.previous_distribution = self

        return new_dist

    def death(self):

        empties = [i for i in self.Data_Allocation.allocation.keys() if self.Data_Allocation.allocation[i] == []]

        print('Empties: {}'.format(empties))
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
            newcomps.append(self.Gaussian_Mix_Model.components[i][:2] +[new_wts[i]])


        newmodel = Gaussian_Mix_Model(newcomps)
        new_dist = GMM_Distribution(newmodel, self.Data_Allocation, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                        self.ep_k)

        return new_dist

    def mu_gibbs_update(self):

        newcomps = []
        for i in self.Gaussian_Mix_Model.components:
            s_i = sum(self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)])
            n_i = len(self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)])
            newcomps.append([norm.rvs((self.zeta*self.kappa+i[1]*s_i)/(i[1]*n_i+self.kappa), (i[1]*self.kappa)/(self.kappa+ n_i*i[1])), i[1], i[2]])

        newmodel = Gaussian_Mix_Model(newcomps)
        new_dist = GMM_Distribution(newmodel, self.Data_Allocation, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                    self.ep_k)
        new_dist.order_components()

        return new_dist
    def var_gibbs_update(self):

        newcomps = []
        for i in self.Gaussian_Mix_Model.components:
            idat = self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)]
            n_i = len(idat)
            var_i = sum([(j - i[0])**2 for j in idat])
            newcomps.append([i[0], 1/(gamma.rvs(self.alpha+(n_i/2), scale = (1/(self.beta + var_i)))), i[2]])

        newmodel = Gaussian_Mix_Model(newcomps)
        new_dist = GMM_Distribution(newmodel, self.Data_Allocation, self.la, self.delta, self.alpha, self.g, self.ep_h,
                                    self.ep_k)
        return new_dist

    def beta_update(self):

        new_dist = copy.deepcopy(self)
        sumvar = sum([i[1] for i in self.Gaussian_Mix_Model.components])
        new_dist.beta = gamma.rvs(self.alpha*self.kappa+self.g, scale = (1/self.h + sumvar))

        return new_dist

    def allocation_update(self):
        newdict = self.Gaussian_Mix_Model.allocate_data(self.Data_Allocation.all_data())[0]

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
            return n_4.beta_update()
        else:
            return n_4

    def continuous_forward_eval(self):

        wtprob = dirichlet.logpdf(self.Gaussian_Mix_Model.wts, np.array([self.delts[i] + len(self.Data_Allocation.allocation[i]) for i in self.Data_Allocation.allocation]))
        muprob = 0

        for i in self.Gaussian_Mix_Model.components:
            s_i = sum(self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)])
            n_i = len(self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)])
            muprob += norm.logpdf(i[0], (self.zeta*self.kappa+i[1]*s_i)/(i[1]*n_i+self.kappa), (i[1]*self.kappa)/(self.kappa+ n_i*i[1]))

        varprob = 0

        for i in self.Gaussian_Mix_Model.components:
            idat = self.Data_Allocation.allocation[self.Gaussian_Mix_Model.components.index(i)]
            n_i = len(idat)
            var_i = sum([(j - i[0])**2 for j in idat])
            varprob -= gamma.logpdf(i[1], self.alpha+(n_i/2), scale = (1/(self.beta + var_i)))

        return wtprob + muprob + varprob



    def discrete_forward_sample(self, move_pmf=[0.5, 0, 0.5], disc_pmf=[0.5, 0.5]):

        discrete_moves = [['split', 'birth'], ['merge', 'death']]

        move_choice = util.assign_from_pmf(move_pmf)
        move = util.assign_from_pmf(disc_pmf)
        if move_choice == 0:
            if move == 0:
                print('merging')
                return self.merge()
            else:
                print('killing')
                return self.death()
        elif move_choice == 0:
            return self.continuous_forward_sample()
        else:
            if move == 0:
                print('Splitting')
                return self.split()
            else:
                print('Making up')
                return self.birth()

    def split_log_eval(self, previous, split_prob):
        """

        Returns: Tuple whose first entry is the probability of all continuous parameters and whose second is the jump probability
        """
        #check if current distribution could have been derived from previous distribution by a split at all
        if self.Gaussian_Mix_Model.k - previous.Gaussian_Mix_Model.k != 1:
            return 0
        elif self.Gaussian_Mix_Model.k > 2 and util.matchlist(self.Gaussian_Mix_Model.means, previous.Gaussian_Mix_Model.means) != 2:
            return 0
        #if previous distribution is compatible, find the  split index
        else:
            ind = 0
            for i in range(self.Gaussian_Mix_Model.k):
                if self.Gaussian_Mix_Model.components[i] == previous.Gaussian_Mix_Model.components[i]:
                    ind+=1
                else:
                    break
            print('Split index: {}'.format(ind))
            #if ind == self.Gaussian_Mix_Model.k:
                #ind+=1

            #isolate the split component data from current and previous
            splits = [self.Gaussian_Mix_Model.components[ind], self.Gaussian_Mix_Model.components[ind + 1]]
            sc = previous.Gaussian_Mix_Model.components[ind]

            # compute selected auxiliary random variables and their probability
            u_1 = splits[0][2] / sc[2]
            u_2 = ((splits[1][0] - sc[0]) / (np.sqrt(sc[1]))) * np.sqrt(splits[1][2] / splits[0][2])
            u_3 = (splits[0][1]*splits[0][2])/((1-u_2**2)*sc[1]*sc[2])
            us = [u_1, u_2, u_3]
            print('us:{}'.format(us))
            uprob = beta.logpdf(u_1, 2, 2)+beta.logpdf(u_2, 2, 2)+beta.logpdf(u_3,1,1)

            #compute the alloction probability of the split
            palloc = 0
            normed_split_weights = util.normalise([splits[0][2], splits[1][2]])
            for i in self.Data_Allocation.allocation[ind]:
                palloc+=math.log(normed_split_weights[0]) + norm.logpdf(i, splits[0][0], np.sqrt(splits[0][1]))
            for i in self.Data_Allocation.allocation[ind+1]:
                palloc+=math.log(normed_split_weights[1]) + norm.logpdf(i, splits[1][0], np.sqrt(splits[1][1]))

            #compute probabiility of choosing split component values
            log_mueval = norm.logpdf(splits[1][0], self.kappa) + norm.logpdf(splits[0][0], self.kappa)
            log_vareval = gamma.logpdf(splits[1][1], self.alpha, scale=1 / self.beta) + gamma.logpdf(splits[0][1],
                                                                                                     self.alpha,
                                                                                                     scale=1 / self.beta)
            l1 = len(self.Data_Allocation.allocation[ind])
            l2 = len(self.Data_Allocation.allocation[ind + 1])
            log_wteval = (self.delta - 1 + l1) * math.log(splits[0][2]) + (self.delta - 1 + l2) * math.log(splits[1][2]) - math.log(betfunc(self.delta, previous.Gaussian_Mix_Model.k*self.delta))

            #compute jacobian of split function
            log_J = math.log(sc[2]) +math.log(np.abs(splits[1][0] - splits[0][0]))  +math.log(splits[1][1]) +math.log(splits[0][1]) - math.log(
                sc[1]) +math.log(1 - (us[1] ** 2)) +math.log(us[2]) + math.log(1 - us[2])

            q_x = math.log(self.Gaussian_Mix_Model.k, self.la) + poisson.logpmf(
                self.Gaussian_Mix_Model.k, self.la)  + log_wteval + log_vareval + log_mueval + log_J

            r_x = uprob + palloc + math.log(split_prob)

            return q_x, r_x

    def merge_log_eval(self, previous, mergeprob):

        # check if current distribution could have been derived from previous distribution by a split at all
        if previous.Gaussian_Mix_Model.k - self.Gaussian_Mix_Model.k != 1:
            return 0
        elif previous.Gaussian_Mix_Model.k > 2 and util.matchlist(previous.Gaussian_Mix_Model.means,
                                           self.Gaussian_Mix_Model.means) != 2:
            return 0
        # if previous distribution is compatible, find the  merge index
        else:
            ind = 0
            for i in range(previous.Gaussian_Mix_Model.k):
                if previous.Gaussian_Mix_Model.components[i] == self.Gaussian_Mix_Model.components[i]:
                    ind += 1
                else:
                    break

            print('Merged at index {}'.format(ind))

            # isolate the split components from the previous distribution that were merged
            splits = [previous.Gaussian_Mix_Model.components[ind], previous.Gaussian_Mix_Model.components[ind + 1]]
            sc = self.Gaussian_Mix_Model.components[ind]

            #Compute the probability of the new merged component parameters
            log_mueval = norm.logpdf(sc[0], self.kappa)
            log_vareval = gamma.logpdf(sc[1], self.alpha, scale=1 / self.beta)
            l_all = len(self.Data_Allocation.allocation[ind])
            log_wteval = (self.delta - 1 + l_all) * math.log(sc[2])


            # compute hypothetical auxiliary randomness to induce 'reverse split'
            u_1 = splits[0][2] / sc[0]
            u_2 = ((splits[1][0] - sc[0]) / (np.sqrt(sc[1]))) * np.sqrt(splits[1][2] / splits[0][2])
            u_3 = (splits[0][1] * splits[0][2]) / ((1 - (u_2 ** 2)) * sc[2] * sc[1])
            

            us = [u_1, u_2, u_3]
            print('Jacobian us: {}'.format(us))

            #compute the Jacobian
            log_J = math.log(sc[2] * (splits[1][0] - splits[0][0]) * splits[1][1] * splits[0][1]) - math.log(
                sc[1] * (1 - us[1] ** 2) * us[2] * (1 - us[2]))


            q_x =  -(poisson.logpmf(self.Gaussian_Mix_Model.k, self.la) + log_mueval + log_vareval + log_wteval + log_J)
            r_x = -math.log(mergeprob)

            return q_x, r_x

    def birth_log_eval(self, previous, birthprob):

        if self.Gaussian_Mix_Model.k - previous.Gaussian_Mix_Model.k != 1:
            return 0
        elif util.matchlist(self.Gaussian_Mix_Model.means, previous.Gaussian_Mix_Model.means) != 1:
            return 0
            # if previous distribution is compatible, find the  birth index
        else:
            if self.Gaussian_Mix_Model.components[:previous.Gaussian_Mix_Model.k] == previous.Gaussian_Mix_Model.components:
                ind = self.Gaussian_Mix_Model.k-1
            else:
                ind = 0
                for i in range(self.Gaussian_Mix_Model.k):
                    if self.Gaussian_Mix_Model.components[i] == previous.Gaussian_Mix_Model.components[i]:
                        ind += 1
                    else:
                        break
            print('Insertion at index {}'.format(ind))

            sc = self.Gaussian_Mix_Model.components[ind]

            wtprob = math.log(sc[2])*(self.delta-1) + math.log(1-sc[2])*(previous.Gaussian_Mix_Model.k*(self.delta-1)+len(self.Data_Allocation.all_data()))-betfunc(self.delta, (previous.Gaussian_Mix_Model.k)*self.delta)
            compprob = poisson.logpmf(self.Gaussian_Mix_Model.k, self.la)

            q_x = wtprob + compprob
            r_x = math.log(birthprob) + beta.logpdf(sc[2], 1, previous.Gaussian_Mix_Model.k)

        return q_x, r_x

    def death_log_eval(self, previous, deathprob):

        if previous.Gaussian_Mix_Model.k - self.Gaussian_Mix_Model.k != 1:
            return 0
        elif util.matchlist(previous.Gaussian_Mix_Model.means, self.Gaussian_Mix_Model.means) != 1:
            return 0
            # if previous distribution is compatible, find the  birth index
        else:
            ind = 0
            for i in range(self.Gaussian_Mix_Model.k):
                if previous.Gaussian_Mix_Model.components[i] == self.Gaussian_Mix_Model.components[i]:
                    ind += 1
                else:
                    break

        sc = previous.Gaussian_Mix_Model.components[ind]
        k_0 = len(previous.Data_Allocation.get_empties())

        q_x = poisson.logpmf(self.Gaussian_Mix_Model.k, self.la)
        r_x = math.log(deathprob) - math.log(k_0 + 1) + math.log(sc[2])**self.Gaussian_Mix_Model.k

        return q_x, r_x
    def discrete_forward_eval(self, splitprob, mergeprob, birthprob, deathprob):

        if self.last_move == 'split':

            return self.split_log_eval(self.previous_distribution, splitprob)

        elif self.last_move == 'birth':
            print('Evaluating birth')

            return self.birth_log_eval(self.previous_distribution, mergeprob)

        elif self.last_move == 'merge':

            return self.merge_log_eval(self.previous_distribution, birthprob)

        elif self.last_move == 'death':

            return self.death_log_eval(self.previous_distribution, deathprob)

        else:
            return 0, 0





