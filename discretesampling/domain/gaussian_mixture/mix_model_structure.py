import numpy as np
import math
import copy
from bisect import bisect
from discretesampling.base.types import DiscreteVariable
from discretesampling.domain.gaussian_mixture.mix_model_distribution import UnivariateGMMProposal
from discretesampling.domain.gaussian_mixture.mix_model_target import UnivariateGMMTarget
import discretesampling.domain.gaussian_mixture.util as gmm_util
import discretesampling.domain.gaussian_mixture.mix_model_initial_proposal as mmi
from scipy.special import logsumexp
from scipy.stats import invgamma


class AllocationStructure:
    """
    Separate class which keeps the actual data for computing allocations
    """
    def __init__(self, h_epsilon, data, initial_gmm):
        self.data = data
        self.allocation = {k:[] for k in initial_gmm.indices}
        self.log_palloc = 0

        self.zeta = np.median(self.data)
        self.data_range = max(self.data) - min(self.data)
        self.kappa = 1 / self.data_range ** 2
        self.h_epsilon = h_epsilon
        self.h = h_epsilon / self.data_range ** 2

        self.propose_allocation(initial_gmm)

    def propose_allocation(self, gmm):
        """
        Parameters
        ----------

        gmm: Univariate GMM to which data will be allocated

        Returns
        -------
        Updated allocation dictionary according to the parameters of the input gmm
        """
        data_allocations = {ind:[] for ind in gmm.indices}
        log_palloc = 0
        indices = gmm.indices
        for i in range(len(self.data)):
            log_prob_alloc = []
            for j in indices:
                try:
                    logp = (self.data[i] - gmm.means[j])**2 / (2 * gmm.covs[j])
                except OverflowError:
                    logp = None

                fac = math.log(gmm.compwts[j]) - (0.5*math.log(gmm.covs[j]))
                if logp is None:
                    log_prob_alloc.append(fac)
                else:
                    log_prob_alloc.append(fac - logp)

            prob_alloc = np.exp(log_prob_alloc - logsumexp(log_prob_alloc))
            prob_cdf = np.cumsum(gmm_util.normalise(prob_alloc))
            q = np.random.uniform(0, 1)

            try:
                comp_index = gmm_util.find_rand(prob_cdf, q)
            except:
                raise Exception('Cannot compute probabilities at logprob {}, cumprob {}'.format(prob_alloc, prob_cdf))
            log_palloc += log_prob_alloc[comp_index]
            data_allocations[comp_index].append(self.data[i])

        self.allocation = data_allocations
        self.log_palloc = log_palloc

    def get_counts(self):
        return [len(self.allocation[k]) for k in self.allocation]

    def get_empties(self):
        return [k for k in list(self.allocation.keys()) if len(self.allocation[k]) == 0]

    def birth_allocation(self, index):
        #print('Insertion at {}'.format(index))
        new_dict = {index:[]}
        for i in self.allocation:
            if i < index:
                new_dict[i] = self.allocation[i]
            elif i >= index:
                new_dict[i+1] = self.allocation[i]

        self.allocation = new_dict

    def kill_allocation(self,index):
        if len(self.allocation[index]) !=0:
            return Exception('Component {} has data allocated to it and cannot be erased'.format(index))
        else:
            end = max(list(self.allocation.keys()))
            for i in self.allocation:
                if i > index:
                    self.allocation[i-1] = self.allocation[i]
            del self.allocation[end]

    def split_allocation(self, gmm):
        if gmm.last_move != 'split':
            raise Exception('No split has occurred to allocate!')
        else:
            ind = gmm.last_split[0]
            split_means = [gmm.means[ind], gmm.means[ind+1]]
            split_covs = [gmm.covs[ind], gmm.covs[ind+1]]
            split_wts = gmm_util.normalise([gmm.compwts[ind], gmm.compwts[ind+1]])
            log_palloc = 0
            data_to_allocate = self.allocation[ind]
            new_allocations = [[],[]]
            for i in range(len(data_to_allocate)):
                log_prob_alloc = []
                for j in [0,1]:
                    logp = -(data_to_allocate[i] - split_means[j]) ** 2 / (2 * split_covs[j])
                    fac = math.log(split_wts[j]) - (0.5*math.log(split_covs[j]))
                    log_prob_alloc.append(fac + logp)

                prob_cdf = np.cumsum([math.exp(i) for i in log_prob_alloc])[0]
                q = np.random.uniform(0, 1)

                if q < prob_cdf:
                    new_allocations[0].append(data_to_allocate[i])
                    log_palloc += log_prob_alloc[0]
                else:
                    new_allocations[1].append(data_to_allocate[i])
                    log_palloc += log_prob_alloc[1]

            self.birth_allocation(ind)
            self.allocation[ind] = new_allocations[0]
            self.allocation[ind+1] = new_allocations[1]
            self.log_palloc = log_palloc

    def merge_allocation(self, gmm):
        """

        Parameters
        ----------
        index: integer
        index of component to be merged

        Returns
        -------
        Assigns any data in the index+1th component to the index component
        and updates allocations and empty components accordingly.
        NB if the index+1th component is empty, this is equivalent to a reversal of a death move
        """
        if gmm.last_move != 'merge':
            raise Exception('The previous move was not a merge step')
        else:
            i = gmm.last_merge[0]
            new_allocation = {}
            for k in self.allocation:
                if k == i:
                    new_allocation[k] = self.allocation[k]+self.allocation[k+1]
                elif k < i:
                    new_allocation[k] = self.allocation[k]
                elif k > i+1:
                    new_allocation[k-1] = self.allocation[k]

            self.allocation = new_allocation

class UnivariateGMM(DiscreteVariable):
    def __init__(self, g, alpha, la, delta, means, covs, compwts):

        self.compwts = list(gmm_util.normalise(compwts))
        self.means = means
        self.covs = covs
        self.n_comps = len(self.compwts)
        self.indices = [i for i in range(self.n_comps)]

        'Fixed Hyperparameters'
        self.g = g
        self.alpha = alpha
        self.la = la
        self.delta = delta

        'Mutable Beta'
        self.beta = None

        'Track last action'
        self.last_move = None
        self.last_merge = None
        self.last_birth = None
        self.last_death = None

    def getInitialProposalType(self, data):

        return mmi.UnivariateGMMInitialProposal(self.la, self.g, self.alpha, self.delta, self.h_epsilon, data).return_initial_distribution()

    def getProposalType(self, allocation_structure):

        proposal = UnivariateGMMProposal(self, allocation_structure)

        return proposal

    def getTargetType(self):
        return UnivariateGMMTarget(self.means, self.covs, self.compwts)

    def get_beta_prior(self, data_allocation):
        self.beta = np.random.gamma(self.g, data_allocation.h)

    def insert_component(self, mu, cov, wt):
        """
        Parameters
        ----------
        mu: float
            mean of new component
        cov: float
            covariance of new component
        wt: float
            proposed weight of new component
            (weights will be normalised automatically)


        Returns
        -------
        When run, updates distribution with component inserted and all weights normalised.
        Insertion will be at index required to ensure means are sorted
        No data will be assigned to the new component
        Returns index at which component was inserted.
        """

        insert_point = bisect(self.means, mu)

        self.means.insert(insert_point, mu)
        self.covs.insert(insert_point, cov)
        self.compwts.insert(insert_point, wt)
        self.n_comps = len(self.compwts)
        self.indices = [i for i in range(self.n_comps)]

        for i in range(len(self.compwts)):
            if i != insert_point:
                self.compwts[i] = self.compwts[i]*(1-wt)
            elif i > insert_point:
                self.compwts[i] = self.compwts

        self.compwts = list(gmm_util.normalise(self.compwts))

        return insert_point

    def remove_component(self, index):
        """
        Parameters
        ----------
        index: integer
                index of the component to be removed
                raises error if data in sample assigned to component

        Returns
        -------
        Distribution with data reallocated to correct component indices
        """

        del (self.means[index])
        del (self.covs[index])
        self.compwts = list(self.compwts)
        del (self.compwts[index])

        self.compwts = list(gmm_util.normalise(self.compwts))

        self.n_comps = len(self.compwts)
        self.indices = [i for i in range(len(self.compwts))]

    def split(self, allocation_structure):
        """
        Parameters
        ----------
        allocation_structure: AllocationStructure type
        Required to generate split hyperparameters and to update proposed allocations

        Returns
        -------
        Proposal distribution with randomly selected peak split into two
        Allocation Structure has proposal allocations updated with data from split component
        reallocated randomly by weight with proposal logprob updated with allocation probability

        Updates last_move attribute of current distribution with 'birth'
        Updates last_split attribute with:
        [
        Index of first peak in last split
        [mean, covariance and weight of split peak]
        [random u_i variables that determined the split (see Richardson & Green 1997)]
        Probability of this particular allocation, returned by update_allocations function
        ]
        """
        prop = copy.deepcopy(self)
        prop_alloc = copy.deepcopy(allocation_structure)

        split_index = np.random.choice(self.indices)

        u_1 = np.random.beta(2, 2)
        u_2 = np.random.beta(2, 2)
        u_3 = np.random.beta(1, 1)

        w_1 = self.compwts[split_index]*u_1
        w_2 = self.compwts[split_index]*(1-u_1)

        s_1 = u_3*(1-u_2**2)*self.covs[split_index]*(self.compwts[split_index]/w_1)
        s_2 = (1-u_3) * (1 - u_2**2) * self.covs[split_index] * (self.compwts[split_index] / w_2)

        mu_1 = self.means[split_index]+(u_2*np.sqrt(self.covs[split_index])*np.sqrt(w_2/w_1))
        mu_2 = self.means[split_index]-(u_2*np.sqrt(self.covs[split_index]) * np.sqrt(w_1/w_2))

        newmeans = [mu_1, mu_2]
        if gmm_util.check_ordered(newmeans):
            newcovs = [s_1, s_2]
            new_wts = [w_1, w_2]
        else:
            newmeans = sorted(newmeans)
            newcovs = [s_2, s_1]
            new_wts = [w_2, w_1]

        test_means = gmm_util.insert_pair(copy.deepcopy(prop.means), newmeans, split_index)

        if gmm_util.check_ordered(test_means):
            prop.means = test_means
            prop.covs = gmm_util.insert_pair(prop.covs, newcovs, split_index)
            prop.compwts = list(gmm_util.normalise(gmm_util.insert_pair(prop.compwts, new_wts, split_index)))
            prop.indices = [i for i in range(len(prop.compwts))]
            prop.n_comps = len(self.compwts)

            prop.last_move = 'split'
            prop.last_split = [split_index, [self.means[split_index], self.covs[split_index], self.compwts[split_index]],[u_1, u_2, u_3]]
            prop_alloc.split_allocation(prop)

            self.last_move = 'merge'
            self.last_merge = [split_index, [newmeans, newcovs, new_wts], [u_1, u_2, u_3]]


        else:
            #print('Not ordered!')
            prop.last_move = 'split_rejected'

        #print('Updated means'.format(prop.means))
        return prop, prop_alloc

    def merge(self, allocation_structure):
        """
        Parameters
        ----------
        allocation_structure: AllocationStructure type


        Returns
        -------
        Randomly selects a peak and merges that peak with the one with the next largest mean
        All data allocated to those two peaks are reallocated to the merged peak
        Updates last_merge data with:
        [
        Index of peak which has been merged
        [means, covariances and weights of original two peaks as two element lists]
        ]
        """
        prop = copy.deepcopy(self)
        prop_alloc = copy.deepcopy(allocation_structure)

        if len(prop.means) > 1:
            merge_index = np.random.choice(self.indices[:len(self.indices)-1])
            #print('Merging at index {}'.format(merge_index))
            merge_wts = [self.compwts[merge_index], self.compwts[merge_index + 1]]
            merge_covs = [self.covs[merge_index], self.covs[merge_index + 1]]
            merge_means = [self.means[merge_index], self.means[merge_index + 1]]

            prop.compwts[merge_index] = sum(merge_wts)
            prop.means[merge_index] = (merge_wts[0]*merge_means[0] + merge_wts[1]*merge_means[1])/sum(merge_wts)
            comp1 = merge_wts[0]*(merge_covs[0]+merge_means[0]**2)
            comp2 = merge_wts[1]*(merge_covs[1]+merge_means[1]**2)
            extract_comp = sum(merge_wts)*prop.means[merge_index]**2
            final = (comp1+comp2-extract_comp)/sum(merge_wts)
            prop.covs[merge_index] = final
            prop.compwts = list(gmm_util.normalise(prop.compwts))

            prop.n_comps = len(self.compwts)
            del (prop.indices[-1])
            prop.remove_component(merge_index+1)

            u_1 = min(1, merge_wts[0]/sum(merge_wts))
            a = np.abs(prop.means[merge_index] - merge_means[0])
            try:
                b = 0.5*(math.log(prop.covs[merge_index]) + math.log(merge_wts[1]) - math.log(merge_wts[0]))
                u_2 = min(1,a/(math.exp(b)))
            except:
                print('Problem with {}'.format(prop.covs[merge_index]))
                u_2 = 0

            u_3 = (merge_covs[0]*merge_wts[0])/(prop.covs[merge_index]*prop.compwts[merge_index]*(1-(u_2**2)))

            prop.last_move = 'merge'
            prop.last_merge = [merge_index, [merge_means, merge_covs, merge_wts], [u_1, u_2, u_3]]

            prop_alloc.merge_allocation(prop)

            self.last_move='split'
            self.last_split = [merge_index, [prop.means[merge_index], prop.covs[merge_index], prop.compwts[merge_index]],[u_1, u_2, u_3]]

        else:
            prop.last_move = 'merge_rejected'

        return prop, prop_alloc

    def birth(self, allocation_structure):
        """
        Creates a new Gaussian component

        Parameters
        ----------
        allocation_structure: AllocationStructure type
        required to generate parameters for birth location (including beta update)


        Returns
        -------
        Returns gmm structure with new component
        Updates data allocations with new empty component
        """
        prop = copy.deepcopy(self)
        prop_alloc = copy.deepcopy(allocation_structure)

        if self.beta is None:
            self.beta = self.get_beta_prior(allocation_structure)

        new_wt = np.random.beta(1, len(self.compwts))
        new_mu = np.random.normal(allocation_structure.zeta, np.sqrt(1/allocation_structure.kappa))
        new_cov = 1/np.random.gamma(self.alpha, self.beta)
        ind = prop.insert_component(new_mu, new_cov, new_wt)

        prop_alloc.birth_allocation(ind)

        prop.last_move = 'birth'
        prop.last_birth = ind

        self.last_move = 'death'
        self.last_death = ind

        return prop, prop_alloc

    def death(self, allocation_structure):
        """
        Parameters
        ----------
        allocation_structure: AllocationStructure type
        Required to check if there are any currently empty allocations

        Returns
        -------
        Proposal structure with empty component removed and component weights renormalised
        Current last_death attribute udpated with list containing mean, covariance and weight of killed component.
        Populates allocation proposal with weights above index reallocated, with logprob = 0

        """

        prop = copy.deepcopy(self)
        prop_alloc = copy.deepcopy(allocation_structure)
        empties = allocation_structure.get_empties()

        if empties == [] or len(self.means) == 1:
            allocation_structure.proposed_allocations = allocation_structure.allocation
            allocation_structure.proposed_logprob = 0
            prop.last_move = 'death_rejected'

        else:
            death_index = np.random.choice(empties)
            #print('Killing index {}'.format(death_index))
            prop.remove_component(death_index)

            prop.last_move = 'death'
            prop.last_death = death_index
            prop_alloc.kill_allocation(death_index)

            self.last_move = 'birth'
            self.last_birth = death_index

        return prop, prop_alloc

    def update_weights(self, allocation_structure):
        """

        Returns
        -------
        Proposal distribution with weights updated according current data allocation
        """
        prop = copy.deepcopy(self)
        counts = allocation_structure.get_counts()

        new_dirichlet = [(self.delta + i) for i in counts]
        prop.compwts = list(np.random.dirichlet(new_dirichlet))

        self.last_move = 'weights'
        return prop, allocation_structure

    def update_parameters(self, allocation_structure):
        """
        Returns
        -------
        Distribution with updated means and covariances as per Richardson & Green 1997
        """
        prop = copy.deepcopy(self)
        kappa = allocation_structure.kappa
        zeta = allocation_structure.zeta
        if self.beta is None:
            self.beta = self.get_beta_prior(allocation_structure)

        new_means = []
        new_covs = []
        all_counts = allocation_structure.get_counts()
        alloc_sums = []
        meandiff_sums = []
        meandiff_avs = []

        for i in self.indices:
            alloc_sum = sum(allocation_structure.allocation[i])
            meandiff_sum = 0
            meandiff_av = 0
            for j in allocation_structure.allocation[i]:
                meandiff_sum += self.means[i]-j
                meandiff_av += (self.means[i]-j)**2

            alloc_sums.append(alloc_sum)
            meandiff_sums.append(meandiff_sum)
            if allocation_structure.allocation[i]:
                meandiff_avs.append(meandiff_av/len(allocation_structure.allocation[i]))
            else:
                meandiff_avs.append(0)

        'Compute new covariances'
        for i in self.indices:
            #print('Perturbing var {}={}'.format(i, self.covs[i]))

            cov_shape= self.alpha + all_counts[i]
            cov_scale = self.beta + meandiff_avs[i] + ((allocation_structure.kappa*all_counts[i]*(meandiff_sums[i]**2)))/(allocation_structure.kappa+all_counts[i])
            #print('Var Parameters are {},{}'.format(cov_shape, cov_scale))
            r = invgamma.rvs(cov_shape, scale=cov_scale)
            new_covs.append(r)
            #print('New var {}={}'.format(i, new_covs[-1]))

        inv_covs = [i**-1 for i in new_covs]
        #print('New covariances: {}'.format(new_covs))

        'compute new covariances from updated means'
        for i in self.indices:
            #print('Perturbing mean {}={}'.format(i, self.means[i]))
            mu_scale = ((allocation_structure.kappa*allocation_structure.zeta) + alloc_sums[i])/(all_counts[i]+allocation_structure.kappa)
            mu_shape = new_covs[i]/(allocation_structure.kappa+all_counts[i])
            #print('Mean Parameters are {},{}'.format(mu_scale, np.exp(mu_shape)))
            new_means.append(np.random.normal(mu_scale, mu_shape))

            #print('New mean {}={}'.format(i, new_means[-1]))

        if gmm_util.check_ordered(new_means):
            prop.means = new_means
            prop.covs = new_covs
            self.last_move = 'parameters'
        else:
            self.last_move = 'parameters_rejected'

        return prop, allocation_structure

    def update_beta(self, allocation_structure):
        """
        Updates parameter governing potential range of covariances of new components
        See Richardson & Green 1997 for details

        Returns
        -------
        Updates beta parameter of distribution, which will be applied to any subsequent parameter updates.
        """

        #print('Using covariances: {}'.format(self.covs))
        inv_covs = [i**-1 for i in self.covs]
        #print('Inverse covariances are {}'.format(inv_covs))

        try:
            self.beta = np.random.gamma(self.g + (self.n_comps * self.alpha), sum(inv_covs) + allocation_structure.h)
            self.last_move = 'beta'
        except:
            raise Exception('Issue with input parameters: {},{},{}'.format((self.n_comps * self.alpha), sum(inv_covs), allocation_structure.h))

        return self, allocation_structure

    def propose_allocation_update(self, allocation_structure):
        allocation_structure.propose_allocation(self)

        self.last_move = 'allocation'

        return self, allocation_structure

