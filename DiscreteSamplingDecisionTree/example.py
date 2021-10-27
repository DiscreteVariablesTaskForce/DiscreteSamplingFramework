# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:18:08 2021

@author: efthi
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from create_tree import Tree
import copy
from Tree_sample import TreeDistribution
import random
from Metrics import predict, accuracy
data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=5)


#initialise the tree
node, leafs, features, thresholds =  Tree.initialise_tree(X_train)

sampleTree = [node, leafs, features, thresholds] 
currentTree = copy.deepcopy(sampleTree)

a = 0.2 #grow likelihood(the biger the a the more possibility of growing the tree)
b = 1 #the bigger the b the less number of terminal nodes we have

for i in range(5000):
    current_tree_target, predict_possibilities_current = TreeDistribution.eval(X_train, y_train, sampleTree, a, b)
    
    forward_probability, reverse_probability = TreeDistribution.sample(X_train, sampleTree)
    print("old_tree: ", currentTree)
    print("new_tree: ", sampleTree)
    new_tree_target, predict_possibilities_new = TreeDistribution.eval(X_train, y_train, sampleTree, a, b)
    
    targetRatio = new_tree_target - current_tree_target
    
    proposalRatio = reverse_probability - forward_probability
    
    acceptProbability = min(1, targetRatio + proposalRatio)
    
    
    q= random.random()
    
    if (acceptProbability > q):
        currentTree = copy.deepcopy(sampleTree)
        predict_possibilities_current = predict_possibilities_new
        print("accepted")
    else:
        sampleTree = copy.deepcopy(currentTree)
        print("rejected")
    


    
labels = predict(X_test, currentTree, predict_possibilities_current)
predictive_accuracy = accuracy(y_test, labels)
print(predictive_accuracy)
        

    