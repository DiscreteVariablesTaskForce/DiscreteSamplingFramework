import math
import numpy as np
import pandas as pd
import time
import copy

from numpy.random import seed, randn, rand, randint, shuffle
from functools import partial
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
# from line_profiler import LineProfiler
from functools import lru_cache  # prefer diskcache because it can memoize unhashable types
import precompute
from proxies import *
import hashlib  # for repeatable hashes

# caching support for tall data example (many datapoints per likelihood)
import pickle
from diskcache import Cache  # for saving PF runs
tallcache = Cache('TALL_CACHE', size_limit=int(5e10))

log_root_2pi =  np.log(np.sqrt(2.0 * np.pi))  # the optional constant for 1D Gaussian likelihood (not needed)

def safelogsumexp(ll, axis=-1):  # scipy logsumexp fails with underflow: this fixes that
    amax = ll.max(axis=axis, keepdims=True)
    cll = (ll - amax).clip(min=-50.)
    return (amax + np.log(np.sum(np.exp(cll))))

class UserFn: # GENERIC template for user function
    def __init__(self, proposal, additive=True):
        self.additive = additive  # user can override
        self.counter = 0  # keeps track of term evaluations
        self.total_counter = 0  # this includes cached ones
        self.proposal = proposal  # user provides a proposal function

    def sample_initial_state(self, runs=1):  # returns a list of initial states of length runs
        pass  # user must implement - could be a sample from the prior if available      

    def evaluate(self, state, term_index, with_gradient=False):  # with_gradient is for systems like pytorch that can hang a gradient onto the quantities returned
        pass # user must implement (term in additive structure)

    def __call__(self, state, scenarios, with_gradient = False):
        n = len(scenarios)
        terms = [self.evaluate(state, term, with_gradient) for term in scenarios]
        if 'torch' in dir(state):
            sum_f = torch.sum(torch.stack(terms))
        else:
            sum_f = sum(terms) # standard subset eval
        sum_f += self.evaluate_regularisation(state, with_gradient) * n # OPTIONAL, MUST scale with number of scenarios
        self.total_counter += n # lru cache ignores this side effect
        return(sum_f if self.additive else sum_f/n)
        #            
    def evaluate_regularisation(self, state, with_gradient = False):
        return(0.0) # note this is normally log prior per scenario


class StateNumpy():
    def __init__(self, arr):
        self.params = np.array(arr)  # it is ok if arr is a list or already a np array
    def __array__(self, dtype=None, copy=None):  # dispatch mechanism to allow cast to numpy
        return (self.params) 
    def to_numpy(self):
        return (self.__array__()) # a flat numpy representation of the state suitable for proxy vector calcs 
    def logging(self):
        return (self.__array__()) # no need to reformat
    def __hash__(self):
        return (hash(tuple(self.__array__())))  # to support cache [would be faster to return object id if we never modify]
    def __str__(self):
        return (str(self.params))
    def __eq__(self, other):
        return (np.array_equal(self.params, other.params))
    

class UserFnProxy(UserFn):
    def __init__(self, args):
        self.proposal_sigma = args.proposal_sigma  #must exist
        # self.args = copy.deepcopy(args) # avoid keeping a copy of the args: so hash of empty test fn will be repeatable
        self.N = args.NUM_SCENARIOS
        self.reps = getattr(args, 'reps', 1)
        self.proxy_kNN = getattr(args, 'proxy_kNN', 1)  # nearest neighbour count if relevant
        self.proxy_max_distance = getattr(args, 'proxy_max_distance', 5.0 * args.proposal_sigma)  # nearest neighbour count if relevant        
        self.proxy = args.proxy if ('proxy' in dir(args)) else False
        proxyType = eval(args.proxyType) if ('proxyType' in dir(args)) else QuadraticProxy
        print("PROXY IS ENABLED type = " + str(proxyType) + "\n" if self.proxy else "PROXY IS DISABLED\n")
        if self.proxy:
            if 'proxyVerify' in dir(args):
                self.ProxyInstance = proxyType(self, conservative=args.proxyVerify)
            else:
                self.ProxyInstance = proxyType(self)
        # some generic proxy info is held by the user fn
        self.min_ll = np.full(args.NUM_SCENARIOS, np.nan)  # np.inf flags an initial unknown value (always gets replaced)
        self.pset = False  # whether we have a proxy yet
        self.frozen = False
        # create a proposal function ... the only difficulty is how to bind alternatives to proposal_sigma 
        propfn = eval(args.proposal) if 'proposal' in dir(args) else rw_proposal
        proposal = partial(propfn, sigma_prop=args.proposal_sigma)        
        super().__init__(proposal)  # bind the proposal sigma but leave option to pass adaptive params

    def step_proxy(self, top_row, force_refit=False):
        if not self.frozen:  # if we are allowed to change the proxy
            refitted = self.ProxyInstance.step(top_row, force_refit)
            if not self.pset:
                if refitted:
                    self.pset = True
                    print("PROXY NOW SET")
    #
    # before we have a proxy, we can at least track the min observed likelihood per scenario

    def evaluate_tracked(self, state, term_index, with_gradient=False, as_vector=False):
        # scopy = copy.deepcopy(state)
        ll = self.evaluate(state, term_index, with_gradient, as_vector)
        self.min_ll[term_index] = min(np.nan_to_num(self.min_ll[term_index], nan=np.inf), ll)  # replace nans with -inf as input for max
        return(ll)

    def __call__(self, state, scenarios, with_gradient=False, parent_scenarios=None, proxy_only=False, verbose=False):  # can specify specific scenarios for proxy (otherwise ALL)
        n = len(scenarios)
        self.total_counter += n # actual fn evals (e.g. the denominator for cache hit rate)
        if with_gradient: # only gets called for torch modules
            sum_f = torch.sum(torch.stack([self.evaluate_tracked(state, term, with_gradient) for term in scenarios]))
        else:
            if proxy_only and self.proxy and self.pset: # no actual evaluations
                sum_f = self.ProxyInstance.evaluate_proxy(state, (scenarios if (parent_scenarios is None) else parent_scenarios)) 
            else:
                sum_f = sum([self.evaluate_tracked(state, term, with_gradient) for term in scenarios]) # standard subset eval
        if self.proxy and (n < self.N) and self.pset: # skip proxy only if its a full evaluation
            # add proxy for all the other scenarios not in subsample
            # i.e. add (sum of proxy for all scenarios) - (sum of proxy for this subsample)
            if not proxy_only: # if no actual evals, we've already done the proxy
                proxy_sub = self.ProxyInstance.evaluate_proxy(state, scenarios) 
                proxy_total = self.ProxyInstance.evaluate_total_proxy(state) if (parent_scenarios is None) else self.ProxyInstance.evaluate_proxy(state, parent_scenarios)
                sum_f += proxy_total - proxy_sub
        #
        sum_f /= self.reps # consistent across subsets
        #
        # prior - often just make this zero when working with tall data
        log_prior = self.evaluate_regularisation(state, with_gradient) # already scaled down by N
        if log_prior != 0.0:
            effective_n = (self.N if (parent_scenarios is None) else len(parent_scenarios)) if self.proxy else n
            sum_f +=  log_prior * effective_n
        return(sum_f)


class UFGenericPosteriorProxy(UserFnProxy):
    #def __init__(self, data, proposal_sigma, adaptive = False):
    def __init__(self, args, can_precompute = True):
        super().__init__(args)
        self.likelihood = args.likelihood
        self.prior = self.make_prior(args) # must provide .rvs 
        self.truth = self.prior.sample(seed = 0) # single realisation with same seed every time
        print("New test function with truth = ", self.truth, "\n")
        self.leaf_size = args.LEAF_SIZE if ('LEAF_SIZE' in dir(args)) else 1
        if 'datafile' in dir(args):
            self.data = pd.read_csv(args.datafile).to_numpy()
            if self.data.shape[0] != args.NUM_SCENARIOS:
                raise Exception("Data has wrong shape ", self.data.shape, " ... expected ", args.NUM_SCENARIOS)
        else:
            self.data = self.make_data()
        print("data shape = ", self.data.shape, "should have first dimension ", self.N)
        #self.N = self.data.shape[0] # should be same as NUM_SCENARIOS     
        self.L = self.data.shape[1] # same as self.leaf_size
        self.lockdown_randomness(0)
        self.exact_matches = 0 # keeps track of any exact matches achieved (diagnostics only)
        if (args.Task == 'Disease') and (self.likelihood == 'DiseaseQuick_SEIR') and can_precompute:
            print("Checking for precomputed data")
            prec = precompute.Precomputation(args)
            if not prec.vs is None:
                if prec.vs.shape[0] == self.N:
                    print("USING PRECOMPUTED DISEASE MODEL IN CENTRAL REGION")
                    self.precomp = prec
                    self.precomp_counter = 0
                else:
                    print("NOT ALL SCENARIOS HAVE BEEN PRECOMPUTED", prec.vs.shape)
            else:
                print("NO SCENARIOS HAVE BEEN PRECOMPUTED")
    #
    
    def lockdown_randomness(self, seed): # if likelihood is stochastic
        self.top_level_seed = seed # for unique caching       
    #
    def sample_initial_state(self, runs = 1, **kwargs): # MJAS added defaults so this conforms to HINTS template
        return([self.prior.sample(seed = r + 1) for r in range(runs)]) # reserve seed 0 for true state
    #
    def make_prior(self, args):
        if 'prior' in dir(args):
            if args.prior == 'NIP_Gaussian1D':
                # can be any class with rvs and logpdf
                return(NIP_Gaussian1D())
            if args.prior == 'NIP_LogNormal1D':
                # can be any class with rvs and logpdf
                return(NIP_LogNormal1D())
            if args.prior == 'NIP_Disease':
                # can be any class with rvs and logpdf
                return(PF_disease.NIP_Disease(args))
            if args.prior == 'NIP_Disease_SEIR':
                # can be any class with rvs and logpdf
                return(PF_disease_SEIR.NIP_Disease_SEIR(args))
            else:
                try:
                    return(eval(args.prior + "()"))
                except Exception as e:
                    print(e)
                    raise Exception("UNKNOWN PRIOR??? ", args.prior)
        else:
            return NonInformativePrior(args.dimensions)
    #        
    # create the likelihood function specified in the args
    def make_likelihood(self, state, seed = 0):
        if self.likelihood == 'Normal':
            return(multivariate_normal(mean = state.to_numpy()[0], cov = np.exp(state.to_numpy()[1]))) # not frozen
        elif self.likelihood == 'Disease':
            config = PF_disease.disease_model_defaults
            return PF_disease.DiseaseLikelihood(state, config, seed)
        elif self.likelihood == 'DiseaseQuick':
            return PF_disease.DiseaseQuickLikelihood(state, PF_disease.disease_model_defaults, seed)
        elif self.likelihood == 'Disease_SEIR':
            config = PF_disease_SEIR.disease_model_defaults
            return PF_disease_SEIR.DiseaseLikelihood(state, config, seed)
        elif self.likelihood == 'DiseaseQuick_SEIR':
            config = PF_disease_SEIR.disease_model_defaults
            config['reps'] = self.reps
            return PF_disease_SEIR.DiseaseQuickLikelihood_SEIR(state, config, seed)
        elif self.likelihood == 'FromState':
            return(state) # this can just be the state if it has a logpdf()
    #
    def make_data(self, dataset_seed = 0): # default is to generate synthetic data
        # user could override this to load the data from file
        true_likelihood = self.make_likelihood(self.truth)
        return(true_likelihood.sample(size = [self.N, self.leaf_size], seed = dataset_seed))
        # NB likelihood function may ignore LEAF_SIZE if it's 1 (not relevant) ... it's only >1 for efficiency of block evals, giving 1 extra data dim
    #
    def evaluate_regularisation(self, state, with_gradient = False):
        return(self.prior.logpdf(state.to_numpy())/self.N)
    #
    def evaluate(self, state, term_index, with_gradient = False, as_vector = False):
        cstate = state if with_gradient else copy.deepcopy(state) # if with gradient, we assume state will not be mutated to create proposals
        return(self.cached_eval_fast(cstate, term_index, with_gradient, as_vector, seed = (self.top_level_seed * self.N) + term_index)) # cache only keeps pointer to objects; deepcopy ensures immutable
    # state must be hashable to use the cache
    # note while this may be faster without cache, the computation count would be wrong
    @lru_cache(maxsize = 10000000) 
    def cached_eval_fast(self, state, term_index, with_gradient = False, as_vector = False, seed = 0):# simple args so caching works through assigned/returned states
        counter_increment = 1
        # check we don't have a Nearest Neighbour proxy exact hit: NB proxy stores all scenarios together
        if self.proxy:
            if self.pset:
                if self.ProxyInstance.is_exact(state):
                    self.exact_matches += 1
                    counter_increment = 0
        self.counter += counter_increment
        # at this point we can use an in-memory bypass (precomputed grid)
        if 'precomp' in dir(self):
            logpdf = self.precomp.eval_fast(state, term_index) # as_vector not supported here
            if not math.isnan(logpdf): # NB NaN == NaN fails
                #verify = True
                self.precomp_counter += counter_increment
                return(logpdf)
        this_likelihood = self.make_likelihood(state, seed)
        kwargs = {'gradient':True} if with_gradient else {} # in case gradient param is not supported 
        terms = this_likelihood.logpdf(self.data[term_index], **kwargs)
        #
        if 'torch' in dir(state):
            f = terms.sum()
            if with_gradient:
                f.backward() # attach gdt
            return(f)
        else:    
            return(terms if as_vector else np.sum(terms))