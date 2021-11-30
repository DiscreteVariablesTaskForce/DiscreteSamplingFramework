import numpy as np
import random
from create_tree import Tree
from Test_tree import test
import math
import copy

class TreeDistribution():
    def __init__(self, tree):
       
       self.X_train = tree.X_train
       self.y_train = tree.y_train
       self.tree = copy.deepcopy(tree)
       

    def sample(self):
        #initialise the probabilities of each move
        moves = ["prune", "swap", "change", "grow"]
        moves_prob = [0.4, 0.1, 0.1, 0.4]
        if len(self.tree.tree) == 1:
            moves_prob = [0.0, 0.0, 0.5, 0.5]
        moves_probabilities = np.cumsum(moves_prob)
        random_number= random.random()
        newTree = copy.deepcopy(self.tree)
        if random_number < moves_probabilities[0]:
            #prune
            newTree = newTree.prune()
            
            
    
        elif random_number < moves_probabilities[1]:
            #swap
            newTree = newTree.swap()
            
            
            
        
        elif random_number < moves_probabilities[2]:
            #change
            newTree = newTree.change()
                     
        
        else:
            #grow
            newTree = newTree.grow()
            
        
        return newTree
        
    def eval(self, sampledTree):
        initialTree = self.tree
        moves_prob = [0.4, 0.1, 0.1, 0.4]
        probability = 0
        nodes_differences = [i for i in sampledTree.tree + initialTree.tree if i not in sampledTree.tree or i not in initialTree.tree]
        #In order to get sampledTree from initialTree we must have:
        #Grow
        if (len(initialTree.tree) < len(sampledTree.tree)):
            probability = moves_prob[3] * (1/len(initialTree.X_train[0])) * (1/len(initialTree.X_train[:])) * (1 / len(initialTree.leafs))
        #Prune
        elif (len(initialTree.tree) > len(sampledTree.tree)):
            probability = moves_prob[0] * (1/(len(initialTree.tree) - 1))
        #Change
        elif (len(nodes_differences) == 2):
            probability = moves_prob [2] * (1/len(initialTree.tree)) * 1/len(initialTree.X_train[0]) * 1/len(initialTree.X_train[:])
        #swap
        elif (len(nodes_differences) == 4):
            probability = moves_prob[1] * 1/ ((len(initialTree.tree) * (len(initialTree.tree) -1))/2) 
        
        return probability
            
    
    
def forward(forward, forward_probability):
    forward.append(forward_probability)
    forward_probability = np.sum(forward)
    return forward_probability

def reverse(forward, reverse_probability ):
    reverse_probability = reverse_probability + np.sum(forward)
    return reverse_probability

