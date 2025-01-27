import joblib

import sys
sys.path.append("C:/Users/mattb242/desktop/projects/reversible_jump/local_code/DiscreteSamplingFramework")
sys.path.append("C:/Users/mattb242/desktop/projects/reversible_jump/local_code/SMCComponents")
import discretesampling

from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import straight_SMC_MPI
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import wt_informed_RJSMC_MPI
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import RJMCMC

def read_floats_from_file(filepath):
    floats = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                floateval = [float(value) for value in line.split()]
                floats.extend(floateval)
            except ValueError:
                print(f'Non-float detected in line {line}')

    return floats

testdir = 'C:/Users/mattb242/Desktop/Projects/reversible_jump/results/toy_model'
gal_test_data = read_floats_from_file('C:/Users/mattb242/Desktop/Projects/reversible_jump/test_data/enzyme.txt')
toy_test = Gaussian_Mix_Model([[-8, 1, 0.3], [8, 1, 0.7]])
toy_test_data = toy_test.sample(100)
init_dist = UnivariateGMMInitialProposal(2, 0.2, 2, 1, 1, 10, toy_test_data)

test_SMC =straight_SMC_MPI(init_dist, 32, 1000)

joblib.dump(test_SMC, testdir + '/str_toy_32x1000.gz')