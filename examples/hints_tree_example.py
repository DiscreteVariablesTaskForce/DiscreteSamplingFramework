# HINTS with numpy
#%load_ext line_profiler
import numpy as np
from hints import *
from hints_tree_fn import *
from tqdm import tqdm
from scipy.stats import t
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

from sklearn import datasets
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


# open csv data file
import pandas as pd
# data = pd.read_csv('examples/datasets/abalone.csv')
data = pd.read_csv('datasets/abalone.csv')

# last column is the target
X = data.iloc[:, :-1]
y = data.Target

# we want to use 2048 data points for the HINTS algorithm
NX = 2048
test_size = (data.shape[0] - NX) / data.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=5)
# format the data for data classification (make the last column the target)

# find the nearest factor of 2 to the number of rows

NX = 2 ** int(np.log2(X_train.shape[0]))

# drop the remaining rows

LEAF_SIZE = 32

NUM_SCENARIOS = NX//LEAF_SIZE   
print(NUM_SCENARIOS, "scenarios by", LEAF_SIZE, "in each leaf =", NX, "(dataset size)")

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_known_args()[0] # defaults

if True: # put True in for HINTS, False for simple MCMC where evaluations use all data
    args.levels = 3 #4
    log_branch_factor = 2
    N_0 = 1 # 4 batches of 64 at leaf
    args.design = np.array([N_0] + [2 ** log_branch_factor for l in range(args.levels)])
    NUM_SCENARIOS = N_0 * 2 ** (args.levels * log_branch_factor) # TO DO get from HINTS
    args.iterations = 200
else: #MCMC
    args.levels = 0
    NUM_SCENARIOS = 64
    args.design = np.array([NUM_SCENARIOS])
    args.iterations = 1000 # NB this is not comparable with HINTS iterations ... check the g.counter for actual leaf node (scenario) function evaluations


# design now has levels + 1 entries
args.additive = True # natural for lok likelihoods in Bayesian inference
args.T = 1.0 #top level
args.dT = 0.0

print(args.__dict__)

# parallel runs : everything goes in args
args.LEAF_SIZE = LEAF_SIZE
args.NUM_SCENARIOS = NUM_SCENARIOS

# split the training data by the number of scenarios
X_train_split = np.array_split(X_train, NUM_SCENARIOS, axis=0)
y_train_split = np.array_split(y_train, NUM_SCENARIOS, axis=0)






g = TestFnTree(data, proposal_sigma = args.proposal_sigma)
hmc = HINTS(args, g)
np.random.seed(args.id)

state = g.sample_initial_state(1, 1, 1, runs = 1)[0]
# state  = g.sample_initial_state()
# include initial state in history
history = []
hstate = copy.deepcopy(state)
history.append({'nu':hstate.nu, 'mu':hstate.mu, 'tau':hstate.tau, 'acceptances':copy.deepcopy(hmc.acceptances), 'rejections':copy.deepcopy(hmc.rejections), 'evals_cache':hmc.fn.counter, 'evaluations':hmc.fn.total_counter})   
for tstep in range(args.iterations):
    hmc.shuffle()
    state, correction = hmc.hints(state, args.levels) # e.g. dbg = (t==0)
    # show progress
    if ((tstep%100)==99):
        print(tstep+1, hmc.acceptances, hmc.rejections, hmc.fn.total_counter, hmc.fn.counter)
    #
    hstate = copy.deepcopy(state)
    history.append({'nu':hstate.nu, 'mu':hstate.mu, 'tau':hstate.tau, 'acceptances':copy.deepcopy(hmc.acceptances), 'rejections':copy.deepcopy(hmc.rejections), 'evals_cache':hmc.fn.counter, 'evaluations':hmc.fn.total_counter})   
return(pd.DataFrame(history))