import numpy as np
import copy
from bisect import bisect
from discretesampling.base.types import DiscreteVariable
from discretesampling.domain.gaussian_mixture.mix_model_distribution import UnivariateGMMProposal
from discretesampling.domain.gaussian_mixture.mix_model_target import UnivariateGMMTarget
import discretesampling.domain.gaussian_mixture.util as gmm_util
from scipy.special import logsumexp
from scipy.stats import invgamma


class AllocationStructure:
    """
    Separate class which keeps the actual data and its allocations
    """
    def __init__(self, data, prior):
        self.data = data
        self.current_allocations = None
        self.current_logprob = None
        self.proposed_allocations = None
        self.proposed_logprob = None

        self.zeta = np.median(self.data)
        self.data_range = max(self.data) - min(self.data)
        self.kappa = 1 / self.data_range ** 2
        self.h = 10 / self.data_range ** 2

        self.propose_allocation(prior)

    def clear_current_proposal(self):
        """
        Returns
        -------
        Updates current allocations and log prob with any proposed version
        Then clears proposed allocations and log prob
        """

        if self.proposed_allocations is not None:
            self.current_allocations = copy.deepcopy(self.proposed_allocations)
            self.current_logprob = copy.copy(self.proposed_logprob)
            self.proposed_allocations = None
            self.proposed_logprob = None

    def propose_allocation(self, gmm):
        """
        Parameters
        ----------

        gmm: Univariate GMM to which data will be allocated
        Specific components of the gmm whose data allocations will be updated

        Returns
        -------

        """
        log_palloc = 0
        indices = gmm.indices
        data_allocations = []
        for i in range(len(self.data)):
            log_prob_alloc = []
            for j in indices:
                logp = -(self.data[i] - gmm.means[j]) ** 2 / (2 * gmm.covs[j])
                fac = np.log(gmm.compwts[j]) - np.log(np.sqrt(gmm.covs[j]))
                log_prob_alloc.append(fac + logp)

                prob_alloc = np.exp(log_prob_alloc - logsumexp(log_prob_alloc))
                prob_cdf = np.cumsum(gmm_util.normalise(prob_alloc))
                q = np.random.uniform(0, 1)

                comp_index = gmm_util.find_rand(prob_cdf, q)
                log_palloc += log_prob_alloc[comp_index]
                data_allocations.append(comp_index)

        if self.current_allocations is None:
            print('No allocation yet, proposing new one')
            self.current_allocations = data_allocations
            self.current_logprob = log_palloc
        elif self.proposed_allocations is None:
            print('Adding an allocation proposal')
            self.proposed_allocations = data_allocations
            self.proposed_logprob = log_palloc
        else:
            print('Shifting proposal!')
            self.clear_current_proposal()
            self.proposed_allocations = data_allocations
            self.proposed_logprob = log_palloc

    def propose_allocation_update(self, update_indices, gmm):

        if self.current_allocations is None:
            raise Exception('No existing allocation to update!')
        elif len(gmm.means) == 1:
            self.proposed_allocations = len(self.data)
            self.proposed_logprob = 0
        else:
            log_palloc = 0
            data_allocations = []
            for i in range(len(self.data)):
                if self.current_allocations[i] in update_indices:
                    log_prob_alloc = []
                    for j in update_indices:
                        logp = -(self.data[i] - gmm.means[j]) ** 2 / (2 * gmm.covs[j])
                        fac = np.log(gmm.compwts[j]) - np.log(np.sqrt(gmm.covs[j]))
                        log_prob_alloc.append(fac + logp)

                        prob_alloc = log_prob_alloc - logsumexp(log_prob_alloc)
                        print('Logprobs are {}'.format(prob_alloc))
                        prob_cdf = np.cumsum(gmm_util.normalise([np.exp(i) for i in prob_alloc]))
                        print('Cumulative probability is {}'.format(prob_cdf))
                        q = np.random.uniform(0, 1)

                        comp_index = gmm_util.find_rand(prob_cdf, q)+min(update_indices)
                        log_palloc += log_prob_alloc[comp_index-min(update_indices)]
                        data_allocations.append(comp_index)

                else:
                    data_allocations.append(self.proposed_allocations[i])

        if self.proposed_allocations is not None:
            self.clear_current_proposal()

        self.proposed_allocations = data_allocations
        self.proposed_logprob = log_palloc

    def propose_merged_allocation(self, index):
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
        if self.current_allocations is None:
            raise Exception('No current allocation to merge')
        else:
            proposed_allocations = []
            for i in range(len(self.current_allocations)):
                if self.current_allocations[i] > index:
                    proposed_allocations.append(self.current_allocations[i] -1)
                else:
                    proposed_allocations.append(self.current_allocations[i])

            print('Merged allocation set: {}'.format(set(proposed_allocations)))

        if self.proposed_allocations is not None:
            self.clear_current_proposal()

        self.proposed_allocations = proposed_allocations
        self.logprob = 1

    def insert_empty_allocation(self, index):
        """

        Parameters
        ----------
        index: non-negative integer

        Returns
        -------
        Shifts the allocation of all data at indices greater than input index by +1
        """
        if self.current_allocations is None:
            raise Exception('No current allocation to insert component')
        elif self.proposed_allocations is not None:
            raise Exception('A proposal allocation already exists')
        else:
            self.proposed_allocations = copy.deepcopy(self.current_allocations)
            for i in range(len(self.proposed_allocations)):
                if self.proposed_allocations[i] > index:
                    self.proposed_allocations[i] += 1

    def get_allocation_count(self, gmm, data_allocations):
        """

        Parameters
        ----------
        gmm: univariate gmm structure
        data_allocations: pre-existing set of data allocations,
        taken from the current_allocations or proposed_allocations attribute.

        Returns
        -------
        A list of counts of allocations for all components
        """
        if max(data_allocations) > max(gmm.indices):
            #print('Data allocations: {}, proposal indices {}'.format(set(data_allocations), gmm.indices))
            raise Exception('Some data allocated to non-existent indices: check input structure')
        else:
            allocation_count = []
            for i in gmm.indices:
                c = data_allocations.count(i)
                allocation_count.append(c)

        return allocation_count

    def get_empty_indices(self, gmm, data_allocations):
        """
        Parameters
        ----------
        gmm: Univarite GMM structure
        data_allocations: current list of data allocations

        Returns
        -------
        List of any component indices for which no data is assigned
        """
        if max(data_allocations) > max(gmm.indices):
            raise Exception('Some data allocated to non-existent indices: check input structure')
        else:
            empty_indices = []
            for i in gmm.indices:
                if data_allocations.count(i) == 0:
                    empty_indices.append(i)

            return empty_indices


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
        self.last_action = None
        self.last_split = None
        self.last_merge = None
        self.last_birth = None
        self.last_death = None

    def getProposalType(self):

        proposal = UnivariateGMMProposal(self.means, self.covs, self.compwts, self.g, self.alpha, self.la, self.delta)

        proposal.last_action = self.last_action
        proposal.last_split = self.last_split
        proposal.last_merge = self.last_merge
        proposal.last_birth = self.last_birth
        proposal.last_death = self.last_death

        return proposal

    def getTargetType(self):
        return UnivariateGMMTarget(self.means, self.covs, self.compwts)

    def get_beta_prior(self, data_allocation):
        return np.random.gamma(self.g, data_allocation.h)

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
            prop.indices.append(len(self.indices))
            prop.n_comps = len(self.compwts)

            self.last_action = 'split'
            self.last_split = [split_index, [self.means[split_index], self.covs[split_index], self.compwts[split_index]], [u_1, u_2, u_3]]

            if allocation_structure.current_allocations is not None:
                print('Adding an index at {}'.format(split_index))
                allocation_structure.insert_empty_allocation(split_index)
                allocation_structure.propose_allocation_update([split_index, split_index+1],prop)
            else:
                allocation_structure.propose_allocation(prop)

        else:
            print('Not ordered!')
            allocation_structure.proposed_allocations = allocation_structure.current_allocations
            allocation_structure.proposed_logprob = 0
            self.last_action = 'split_rejected'

        print('Updated means'.format(prop.means))
        return prop

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

        if len(prop.means) > 1:
            merge_index = np.random.choice(self.indices[:len(self.indices)-1])
            merge_wts = [self.compwts[merge_index], self.compwts[merge_index + 1]]
            merge_covs = [self.covs[merge_index], self.covs[merge_index + 1]]
            merge_means = [self.means[merge_index], self.means[merge_index + 1]]

            prop.compwts[merge_index] = sum(merge_wts)
            prop.means[merge_index] = (merge_wts[0]*merge_means[0] + merge_wts[1]*merge_means[1])/sum(merge_wts)
            comp1 = merge_wts[0]*(merge_covs[0]+merge_means[0]**2)
            comp2 = merge_wts[1] * (merge_covs[1]+merge_means[1]**2)
            extract_comp = sum(merge_wts)*self.means[merge_index]**2
            final = (comp1+comp2-extract_comp)/sum(merge_wts)
            prop.covs[merge_index] = final
            prop.compwts = list(gmm_util.normalise(prop.compwts))

            prop.n_comps = len(self.compwts)
            del (prop.indices[-1])
            prop.remove_component(merge_index+1)

            u_1 = merge_wts[0]/sum(merge_wts)
            a = prop.means[merge_index] - merge_means[0]
            b = np.sqrt(prop.covs[merge_index]*(merge_wts[1]/merge_wts[0]))
            u_2 = a/b
            c = (1-(u_2**2))*prop.covs[merge_index]*np.sqrt(prop.compwts[merge_index]/merge_wts[0])
            u_3 = merge_covs[0]/c

            self.last_action = 'merge'
            self.last_merge = [merge_index, [merge_wts, merge_covs, merge_wts], [u_1, u_2, u_3]]

            allocation_structure.propose_merged_allocation(merge_index)

        else:
            self.last_action = 'merge_rejected'
            allocation_structure.proposed_allocations = allocation_structure.current_allocations
            allocation_structure.current_logprob = 0

        return prop

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

        if self.beta is None:
            self.beta = self.get_beta_prior(allocation_structure)

        new_wt = np.random.beta(1, len(self.compwts))
        new_mu = np.random.normal(allocation_structure.zeta, np.sqrt(1/allocation_structure.kappa))
        new_cov = 1/np.random.gamma(self.alpha, self.beta)
        ind = prop.insert_component(new_mu, new_cov, new_wt)

        allocation_structure.insert_empty_allocation(ind)
        allocation_structure.proposed_logprob = 0

        self.last_action = 'birth'

        return prop

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
        empties = allocation_structure.get_empty_indices(self, allocation_structure.current_allocations)

        if not empties:
            allocation_structure.proposed_allocations = allocation_structure.current_allocations
            allocation_structure.proposed_logprob = 0
            self.last_action = 'death_rejected'


        else:
            death_index = np.random.choice(empties)
            print('Killing index {}'.format(death_index))
            prop.remove_component(death_index)

            allocation_structure.propose_merged_allocation(death_index)
            allocation_structure.proposed_logprob = 0
            self.last_action = 'death'

        return prop

    def update_weights(self, allocation_structure):
        """

        Returns
        -------
        Proposal distribution with weights updated according current data allocation
        """
        prop = copy.deepcopy(self)

        if allocation_structure.current_allocations is None:
            raise Exception('No data allocations to determine weight update')
        elif allocation_structure.proposed_allocations is None:
            print('Referring to current allocations')
            allocs = allocation_structure.current_allocations
            print('Allocation counts: {}'.format(allocation_structure.get_allocation_count(prop,allocs)))
        else:
            allocation_structure.clear_current_proposal()
            allocs = allocation_structure.current_allocations
            print('Referring to proposed structure')
            print('Allocation counts: {}'.format(allocation_structure.get_allocation_count(prop, allocs)))

        print('Update')

        new_dirichlet = [(self.delta + i) for i in allocation_structure.get_allocation_count(prop, allocs)]
        prop.compwts = list(np.random.dirichlet(new_dirichlet))

        self.last_action = 'weights'
        return prop

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
        all_counts = allocation_structure.get_allocation_count(self, allocation_structure.current_allocations)
        alloc_sums = []
        meandiff_sums = []

        for i in self.indices:
            alloc_sum = 0
            meandiff_sum = 0
            for j in range(len(allocation_structure.data)):
                if allocation_structure.current_allocations[j] == i:
                    alloc_sum += allocation_structure.data[j]
                    meandiff_sum += (allocation_structure.data[j] - self.means[i])**2

            alloc_sums.append(alloc_sum)
            meandiff_sums.append(meandiff_sum)

        for i in self.indices:
            #print('Perturbing var {}={}'.format(i, self.covs[i]))

            cov_shape= self.alpha + (all_counts[i]/2)
            cov_scale = self.beta + (meandiff_sums[i]/2)
            print('Var Parameters are {},{}'.format(cov_shape, cov_scale))
            r = np.log(np.random.gamma(cov_shape, cov_scale))
            new_covs.append(np.exp(-2*r))
            #print('New var {}={}'.format(i, new_covs[-1]))

        inv_covs = [i**-2 for i in new_covs]
        for i in self.indices:
            #print('Perturbing mean {}={}'.format(i, self.means[i]))
            mu_scale = ((inv_covs[i]*alloc_sums[i])+(kappa*zeta))/((inv_covs[i]*all_counts[i]) + kappa)
            mu_shape = np.log(((inv_covs[i]*all_counts[i]) + kappa)**-1)
            print('Mean Parameters are {},{}'.format(mu_scale, np.exp(mu_shape)))
            new_means.append(np.random.normal(mu_scale, np.exp(mu_shape)))

            #print('New mean {}={}'.format(i, new_means[-1]))

        if gmm_util.check_ordered(new_means):
            prop.means = new_means
            for i in range(len(self.compwts)):
                prop.covs[i] = np.random.normal(self.covs[i], 1)
            self.last_action = 'parameters_accepted'
        else:
            self.last_action = 'parameters_rejected'

        allocation_structure.proposed_allocations = allocation_structure.current_allocations
        allocation_structure.proposed_logprob = 0

        return prop

    def update_beta(self, data_allocation):
        """
        Updates parameter governing potential range of covariances of new components
        See Richardson & Green 1997 for details

        Returns
        -------
        Updates beta parameter of distribution, which will be applied to any subsequent parameter updates.
        """

        inv_covs = [1 / i for i in self.covs]
        prop = copy.deepcopy(self)

        prop.beta = np.random.gamma(self.g + (self.n_comps * self.alpha), sum(inv_covs) + data_allocation.h)
        self.last_action = 'beta'

        return prop