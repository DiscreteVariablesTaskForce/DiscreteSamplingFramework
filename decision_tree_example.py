# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""

from discretesampling import decision_tree

from sklearn import datasets
from sklearn.model_selection import train_test_split
import copy
import random
import math
import numpy as np

data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=5)

initialTree = decision_tree.Tree(X_train, y_train)

a = 0.01
b = 5

currentTree = initialTree
forward_probs = []
sampledTrees = []

for i in range (5000):
    forward_proposal = decision_tree.TreeDistribution(currentTree)
    sampleTree = forward_proposal.sample()
    forward_probability = forward_proposal.eval(sampleTree)
    forward_probability = decision_tree.forward(forward_probs, forward_probability)
    
    reverse_proposal = decision_tree.TreeDistribution(sampleTree)
    reverse_probability = reverse_proposal.eval(currentTree)
    reverse_probability = decision_tree.reverse(forward_probs, reverse_probability)
    
    new_tree_target = sampleTree.evaluatePosterior(a,b)
    current_tree_target = currentTree.evaluatePosterior(a,b)
    
    targetRatio = new_tree_target - current_tree_target
    proposalRatio = math.log(reverse_probability) - math.log(forward_probability)
    acceptLogProbability = min(1, math.exp(targetRatio + proposalRatio))
    
    q= random.random()
    if ((acceptLogProbability) > q):
        currentTree = sampleTree
    else:
        currentTree = currentTree
        del forward_probs[-1]
    
    
    sampledTrees.append(copy.deepcopy(currentTree))

labels = decision_tree.stats.predict(currentTree, X_test)
acc = decision_tree.accuracy(y_test, labels)

