# state and proposal
# NB make sure the state is hashable so the cache works
from hints_fn import *

# create the user function for HINTS to work with this

class TestFnTree(UserFn):
    def __init__(self, data, proposal_sigma):
        self.N = data.shape[0]
        self.per_lead = data.shape[1]
        self.data = data
        super().__init__(lambda state, index: proposalT(state, index, proposal_sigma)) # bind the sigma

    def sample_initial_state(self, nu, mu, tau, runs = 1):
        nus = nu + randint(0, 30, runs)
        # make sure any nu value is at least 1
        nus = np.where(nus < 1, 1, nus)
        mus = mu + 0.25 * randn(runs)
        taus = tau + 0.25 * randn(runs)
        return([StateT(nus[i], mus[i], taus[i]) for i in range(runs)])
        # return(StateT(1, 0.0, 1.0)) # start with fattest distro

    def evaluate(self, state, term_index, with_gradient = False):
        return(self.cached_eval_fast(state.nu, state.mu, state.tau, term_index))

    @lru_cache(maxsize = 1000000) 
    def cached_eval_fast(self, nu, mu, tau, term_index):# simple args so caching works through assigned/returned states
        self.counter += 1
        return(t(nu, loc = mu, scale = tau).logpdf(self.data[term_index]).sum())