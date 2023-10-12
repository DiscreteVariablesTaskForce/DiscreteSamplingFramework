import copy
import numpy as np
from scipy.stats import multivariate_normal

from discretesampling.base.random import RNG
import discretesampling.domain.spectrum as spec
from discretesampling.base.stan_model import StanModel
from discretesampling.base.algorithms.continuous.RandomWalk import RandomWalk
from discretesampling.base.reversible_jump import ReversibleJumpProposal, ReversibleJumpInitialProposal, ReversibleJumpParameters, ReversibleJumpTarget


stan_model_path = "examples/stanmodels/mixturemodel.stan"


# What data should be passed to the stan model given discrete variable x?
def data_function(x):
    dim = x.value
    data = [["K", dim], ["N", 20], ["y", [3, 4, 5, 4, 5, 3, 4, 5, 6, 3, 4, 5, 15, 16, 15, 17, 16, 18, 17, 14]]]
    return data


# What params should be evaluated if we've made a discrete move from x to y
# where params is the params vector for the model defined by x
class continuous_proposal():
    def __init__(self):
        # grid search parameters
        self.grid_size = 1000
        self.min_vals = [0]
        self.max_vals = [np.pi]
        self.inds = [0]

        # store these values here so that they can be used when eval() is called (for efficiency, so we don't have to
        # re-compute many expensive logprob evaluations)
        self.current_discrete = None
        self.current_continuous = None
        self.proposed_discrete = None
        self.proposed_continuous = None
        self.forward_logprob = None
        self.to_remove = None

    def sample(self, x, params, y, rng=RNG()):
        dim_x = x.value
        dim_y = y.value

        self.current_discrete = x
        self.proposed_discrete = y
        self.current_continuous = params

        params_temp = copy.deepcopy(params)
        if dim_x > 1:
            theta = params_temp[0:(dim_x-1)]  # k-simplex parameterised as K-1 unconstrained
            mu = params_temp[(dim_x-1):(2*dim_x-1)]
            sigma = params_temp[(2*dim_x-1):len(params_temp)]
        else:
            theta = []  # One component, mixing component is undefined
            mu = params_temp[0:dim_x]
            sigma = params_temp[dim_x:len(params_temp)]

        if (dim_y > dim_x):
            # Birth move
            # Add new components
            n_new_components = dim_y - dim_x

            if dim_x > 0:
                theta_new = rng.randomMvNormal(np.zeros(n_new_components), np.identity(n_new_components))
                mu_new = rng.randomMvNormal(np.zeros(n_new_components), np.identity(n_new_components))
                sigma_new = rng.randomMvNormal(np.zeros(n_new_components), np.identity(n_new_components))

                mvn = multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))
                forward_logprob = mvn.logpdf(theta_new) + mvn.logpdf(mu_new) + mvn.logpdf(sigma_new)
            else:
                # initial proposal
                theta_new = []
                theta_logprob = 0.0
                if n_new_components > 1:
                    theta_new = rng.randomMvNormal(np.zeros(n_new_components-1), np.identity(n_new_components-1))
                    mvn_theta = multivariate_normal(np.zeros(n_new_components-1), np.identity(n_new_components-1))
                    theta_logprob = mvn_theta.logpdf(theta_new)

                mu_new = rng.randomMvNormal(np.zeros(n_new_components), np.identity(n_new_components))
                sigma_new = rng.randomMvNormal(np.zeros(n_new_components), np.identity(n_new_components))

                mvn = multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))

                forward_logprob = theta_logprob + mvn.logpdf(mu_new) + mvn.logpdf(sigma_new)
            self.forward_logprob = forward_logprob

            params_temp = np.concatenate((np.array(theta), theta_new, np.array(mu), mu_new, np.array(sigma), sigma_new))
            self.proposed_continuous = params_temp
        elif (dim_x > dim_y):
            # randomly choose one to remove
            to_remove = rng.randomInt(0, dim_x-2)  # really need to sort out simplex vs. real param indexing
            # birth of component could happen multiple ways (i.e. different n_new_components)
            # so I think the reverse_logprob will only be approximate - seems like there might
            # be a proof for the summed pdf values of the series of Gaussians that we can use?
            mvn = multivariate_normal(0, 1)
            if dim_y > 1:
                theta = np.delete(theta, to_remove)
            else:
                theta = np.array([])

            mu = np.delete(mu, to_remove)
            sigma = np.delete(sigma, to_remove)

            forward_logprob = np.log(1/dim_x)
            self.forward_logprob = forward_logprob
            self.to_remove = to_remove

            params_temp = list(np.concatenate((theta, mu, sigma)))

        self.proposed_continuous = params_temp
        return list(params_temp)

    def eval(self, x, params, y, p_params):
        dim_x = x.value
        dim_y = y.value

        move_logprob = None

        # first check that we're doing an eval for the previous proposal
        if self.current_discrete == x:
            if self.proposed_discrete == y:
                if np.array_equal(self.current_continuous, params):
                    if np.array_equal(self.proposed_continuous, p_params):
                        if (dim_y > dim_x):
                            # Birth move
                            reverse_logprob = 1 / dim_y
                        elif (dim_x > dim_y):
                            # Death move
                            # Find probabilities of sampling parameters of component that has been removed
                            n_new_components = dim_x - dim_y

                            theta_old = params[self.to_remove]
                            mu_old = params[dim_x + self.to_remove]
                            sigma_old = params[2*dim_x + self.to_remove]

                            mvn = multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))
                            reverse_logprob = mvn.logpdf(theta_old) + mvn.logpdf(mu_old) + mvn.logpdf(sigma_old)
                        move_logprob = reverse_logprob - self.forward_logprob

        if move_logprob is None:
            raise Exception("Proposal probability undefined. This can only be called directly after sampling a from the \
                            proposal.")

        return move_logprob


# initialise stan model
model = StanModel(stan_model_path)

params = ReversibleJumpParameters()

discrete_proposal = spec.SpectrumDimensionProposal()
proposal = ReversibleJumpProposal(discrete_proposal)

initial_discrete_proposal = spec.SpectrumDimensionInitialProposal()
initial_proposal = ReversibleJumpInitialProposal(initial_discrete_proposal)

discrete_target = spec.SpectrumDimensionTarget(3, 2)
continuous_target = model
target = ReversibleJumpTarget(discrete_target, continuous_target, data_function)
