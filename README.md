# DiscreteSamplingFramework
Python classes describing discrete variable sampling/proposals/targets

## Requirements
 - Python 3.x
 - numpy
 - scipy
 - scikit-learn (for examples)

### Discrete Variables
Each example of these should, at minimum implement functions:
 - `__init__` - a constructor with some arguments setting up the basic attributes of the variable
 - `getProposalType` - a class method returning the relevant proposal distribution type for this variable type
 - `getInitialProposalType` - a class method returning the relevant initial proposal distribution type for this variable type
- `getTargetType` - a class method returning the relevant target distribution type for this variable type


### Discrete Variable Proposal Distributions
Each example of these should, at minimum implement functions:
 - `__init__` - constructor with a single argument (a DiscreteVariable, x) which will act as the "starting point" for this proposal
 - `eval` - function taking a single argument (a DiscreteVariable, x') that returns the probability of sampling that DiscreteVariable from this proposal (P(x|x'))
 - `sample` - function with no arguments which returns a sample of a DiscreteVariable, x', from this proposal, q (x' ~ q(x))


### Discrete Variable Initial Proposal Distributions
Similarly, distributions of initial proposals should be described.
Each example of these should, at minimum implement functions:
 - `__init__` - constructor, possibly with some extra parameters
 - `eval` - function taking a single argument (a DiscreteVariable, x) that returns the probability of sampling that DiscreteVariable from this proposal (q0(x))
 - `sample` - function with no arguments which returns a sample of a DiscreteVariable, x', from this initial proposal (x ~ q0())

### Discrete Variable Target Distributions
Each example of these should, at minimum implement functions:
 - `__init__` - constructor, possibly with some parameters theta and data D
 - `eval` - function taking a single argument (a DiscreteVariable, x) that returns the log-posterior density logP(D|x,theta) + logP(x|theta)