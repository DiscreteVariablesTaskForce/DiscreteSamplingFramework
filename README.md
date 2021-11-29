# DiscreteSamplingFramework
Python classes describing discrete variable sampling/proposals

## Discrete Variables
Each example of these should, at minimum implement functions:
 - `__init__` - a constructor with some arguments setting up the basic attributes of the variable
 - `getDistributionType` - a class method returning the relevant Distribution type for this variable type

## Discrete Variable Distributions
Each example of these should, at minimum implement functions:
 - `__init__` - constructor with a single argument (a DiscreteVariable, x) which will act as the "starting point" for this distribution
 - `eval` - function taking a single argument (a DiscreteVariable, x') that returns the probability of sampling that DiscreteVariable from this Distribution (P(x|x'))
 - `sample` - function with no arguments which returns a sample of a DiscreteVariable, x', from this distribution, q (x' ~ q(x))
