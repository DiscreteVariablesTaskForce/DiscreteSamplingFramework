# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:28:58 2021

@author: efthi
"""

import random

class Tree():
    
    def initialise_tree(X_train):
        nodes = [0]
        leafs = [1,2]
        thresholds = []
        features = []
        for node in nodes:
            feature = random.randint(0,len(X_train[0])-1)
            threshold = random.randint(0,len(X_train)-1)
            features.append(feature)
            thresholds.append(X_train[threshold,feature])
        
        return nodes, leafs, features, thresholds
            
    def grow(X_train, nodes, leafs, features, thresholds ):
        
        action = "grow"
        print(action)
        #grow tree by just simply populating the individial lists
        nodes.append(min(leafs))
        leafs.remove(min(leafs))
        leafs.append(max(leafs)+1)
        leafs.append(max(leafs)+1)
        leafs.sort()
        
        
        
        '''
        a new threshold and feature must be added to the new parent node as well
        also we have to calculate the possibilities for each leaf as the tree have changed
        '''
        feature = random.randint(0,len(X_train[0])-1)
        threshold = random.randint(0,len(X_train)-1)
        features.append(feature)
        thresholds.append(X_train[threshold,feature])
        
        return nodes, leafs, features, thresholds, action
    
    def prune(nodes, leafs, features, thresholds):
        action = "prune"
        print(action)
        '''
        For example when we have nodes 0,1,2 and leafs 3,4,5,6 when we prune we take the leafs 6 and 5 out, and the
        node 2, now becomes a leaf. To do that we first need to say that the max number from leafs, 6 in our case must
        be used to find its parent node. To do that we substract 2 from the max leaf and then devide by 2((max leaf-2)/2).
        Now the parent node is a leaf node.
        '''
        nodes.remove((max(leafs)-2)/2)
        leafs.append((max(leafs)-2)//2)
        leafs.remove(max(leafs))
        leafs.remove(max(leafs))
        leafs.sort()
        '''
        when pruning we need to delete the feature from the parent node so as to become a leaf node now
        '''        
        del features[-1]
        del thresholds[-1]


        
        return nodes, leafs, features, thresholds, action
    
    def change(X_train, features, thresholds):
        action = "change"
        print(action)
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have chosen 
        and then pick unoformly a node and change their features and thresholds
        '''
        new_feature = random.randint(0,len(X_train[0])-1)
        new_threshold = random.randint(0,len(X_train)-1)
        position_to_change = random.randint(0,len(thresholds)-1)
        thresholds[position_to_change] = X_train[new_threshold, new_feature]        
        features[position_to_change] = new_feature
        
        return features, thresholds, action
    
    def swap(features, thresholds):
        action = "swap"
        print(action)
        '''
        need to swap the features and the threshold among the 2 nodes
        '''
        feature_to_swap1 = random.randint(0,len(thresholds)-1)
        feature_to_swap2 = random.randint(0,len(thresholds)-1)
            
            #in case we choose the same node
        while feature_to_swap1 == feature_to_swap2:
            feature_to_swap2 = random.randint(0,len(thresholds)-1)
        
        
        temporary_feature = features[feature_to_swap1]
        temporary_threshold = thresholds[feature_to_swap1]
            
        features[feature_to_swap1] = features[feature_to_swap2]
        thresholds[feature_to_swap1] = thresholds[feature_to_swap2]

        features[feature_to_swap2] = temporary_feature
        thresholds[feature_to_swap1] = temporary_threshold
        
        return features, thresholds, action
    

        



    


        
        

