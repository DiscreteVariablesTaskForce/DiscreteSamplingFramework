import math
import numpy as np
import matplotlib.pyplot as plt

from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model

toy_test = Gaussian_Mix_Model([[-4, 1, 0.2], [0, 1, 0.3], [4, 1, 0.5]])
toy_test_data = toy_test.sample(100)

init_dist = UnivariateGMMInitialProposal(3, 0.2, 2, 1, 1, 0.1, toy_test_data).get_initial_dist()

def RJMCMC(current, t):
    i = 0
    accept = 0
    k = []
    probs = []
    comps = []
    dat = current.Data_Allocation.all_data()
    while i <= t:
        print('Starting step {}'.format(i))
        proposed_cont = current.continuous_forward_sample()
        proposed = proposed_cont.discrete_forward_sample()

        current_likelihood = sum([current.Gaussian_Mix_Model.eval(i) for i in dat])
        proposed_likelihood = sum([proposed.Gaussian_Mix_Model.eval(i) for i in dat])
        print('proposed likelihood: {}'.format(proposed_likelihood))

        proposed_eval = proposed.discrete_forward_eval(0.25,0.25,0.25,0.25)
        print('discrete likelihood: {}'.format(proposed_eval))

        acc_ratio = (proposed_likelihood - current_likelihood) + (proposed_eval[0] - proposed_eval[1])
        print('Alpha: {}'.format(acc_ratio))
        acc_prob = min([acc_ratio, 0])
        print('Acceptance: {}'.format(acc_prob))

        u = np.random.uniform(0,1)

        if u < math.exp(acc_prob):
            current = proposed
            print('Accepted!')
            accept+=1

        k.append(current.Gaussian_Mix_Model.k)
        probs.append(current.compute_logprob(toy_test_data))
        comps.append(current.Gaussian_Mix_Model.components)

        i+=1
        print('Current acceptance ratio: {}'.format(accept / i))

    return k, probs, comps

test_RJ = RJMCMC(init_dist,1000)
x = [i for i in range(1001)]
plt.plot(x, test_RJ[0])
plt.show()

plt.plot(x, test_RJ[1])
plt.show()