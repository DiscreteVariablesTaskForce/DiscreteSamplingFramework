# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:03:41 2021

@author: efthi
"""
import numpy as np
import random
from create_tree import Tree
from Test_tree import test
import math

class TreeDistribution():
    
    def sample(X_train, current_Tree):
        '''
        current_Tree is a list of lists
        current_Tree[0] = nodes
        current_Tree[1] = leafs
        current_Tree[2] = features
        current_Tree[3] = thresholds
        '''
        
        #initialise the probabilities of each move
        moves = ["prune", "swap", "change", "grow"]
        moves_prob = [0.3, 0.15, 0.15, 0.14]
        moves_probabilities = np.cumsum(moves_prob)
        random_number= random.random()
        
        if random_number < moves_probabilities [0] and len(current_Tree[0]) > 1:
            #prune
            #(T',theta'|T,theta,a)
            forward_probability = moves_prob[0]
            Tree.grow(X_train, current_Tree[0], current_Tree[1], current_Tree[2], current_Tree[3])
            #(T,theta|T',theta',a)
            reverse_probability = moves_prob[0]  * 1/len(X_train[0]) * 1/len(X_train[:])
            
            return forward_probability, reverse_probability

        elif random_number < moves_probabilities[1] and len(current_Tree[0]) > 1:
            #swap
            #(T',theta'|T,theta,a)
            forward_probability = moves_prob[1] * 1/ ((len(current_Tree[0]) * (len(current_Tree[0]) -1))/2)
            Tree.swap(current_Tree[2], current_Tree[3])
            #(T,theta|T',theta',a)
            reverse_probability = moves_prob [1] * 1/((len(current_Tree[0]) * (len(current_Tree[0]) -1))/2)
            
            return forward_probability, reverse_probability
            
        
        elif random_number < moves_probabilities[2]:
            #change
            #(T',theta'|T,theta,a)
            forward_probability = moves_prob [2] * (1/len(current_Tree[0])) * 1/len(X_train[0]) * 1/len(X_train[:])
            Tree.change(X_train, current_Tree[2], current_Tree[3])
            #(T,theta|T',theta',a)
            reverse_probability = moves_prob [2] * (1/len(current_Tree[0])) * 1/len(X_train[0]) * 1/len(X_train[:])
            
            return forward_probability, reverse_probability
        
        else:
            #grow
            #(T',theta'|T,theta,a)
            forward_probability = moves_prob[3] 
            Tree.grow(X_train, current_Tree[0], current_Tree[1], current_Tree[2], current_Tree[3])
            #(T,theta|T',theta',a)
            reverse_probability = moves_prob[0]
            
            return forward_probability, reverse_probability
        
        
    def eval(X_train, y_train, newTree, a, b):
        #call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = test.calculate_leaf_occurences(X_train, y_train, newTree[0], newTree[1], newTree[2], newTree[3])
        #call test tree to calculate  (theta|T)
        target2 = test.features_and_threshold_probabilities(X_train, newTree[2], newTree[3])
        #p(T)
        target3 = test.prior_calculation(newTree[1], a, b)
            
        return (target1*target2*target3) , leafs_possibilities_for_prediction
            
        
        
        
        


    