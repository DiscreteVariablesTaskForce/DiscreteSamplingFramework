# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:28:40 2021

@author: efthi
"""
import collections
import math
import numpy as np


class test():
    #Π(Y_i|T,theta,x_i)
    def calculate_leaf_occurences(X_train, y_train, nodes, leafs, features, thresholds):
        
        '''
        we calculate how many labelled as 0 each leaf has, how many labelled as 1 each leaf has and so on
        '''
        leaf_occurences = []
        k=0 
        for datum in X_train:

            current_node = 0
            
            #make sure that we are not in leafs
            while current_node not in leafs:
                if  datum[features[current_node]] > thresholds[current_node]:
                    current_node = (current_node*2)+2
                else:
                    current_node = (current_node*2)+1
            
            '''
            create a list of lists that holds the leafs and the number of occurences
            for example [[4,1,1,2,2,2][5,1,1,2,2,1][6,1,2,2,1,2][7,2,2,2,1,2,1,2]]
            The first number represents the leaf id number
            '''
            if not any(current_node in x for x in leaf_occurences):
                leaf_occurences.append([current_node])
                leaf_occurences.sort()
             
            for p in range(len(leaf_occurences)):
                if leaf_occurences[p][0] == current_node:
                    leaf_occurences[p].append(y_train[k])
            k+=1

        '''
        we have some cases where some leaf nodes may do not have any probabilities
        because no data point ended up in the particular leaf
        We add equal probabilities for each label to the particular leaf.
        For example if we have 4 labels, we add 0:0.25, 1:0.25, 2:0.25, 3:0.25
        '''
        fix_unvisited_leafs = []
        if len(leaf_occurences) < len(leafs):
            for i in range(len(leaf_occurences)):
                fix_unvisited_leafs.append(leaf_occurences[i][0])
                
    
            for i in range(len(leafs)):    
                if leafs[i] not in fix_unvisited_leafs :
                    leaf_occurences.insert(leafs[i],[leafs[i],0,1,2])
                
        leaf_occurences = sorted(leaf_occurences)
    
        '''
        we then delete the first number of the list which represents the leaf node id.
        '''
        for i in range(len(leaf_occurences)):
            new_list = True
            for p in range(len(leaf_occurences[i])):
                if new_list :
                    new_list = False
                    del leaf_occurences[i][p] 
        '''
        first count the number of labels in each leaf.
        Then create probabilities by normalising those values[0,1]
        '''
        leafs_possibilities = []
        for number_of_leaves in range(len(leaf_occurences)):
            occurrences = collections.Counter(leaf_occurences[number_of_leaves][:])
            leafs_possibilities.append(occurrences)
            
        #create leafs possibilities
        for item in leafs_possibilities:
            factor=1.0/sum(item.values())
            for k in item:
                item[k] = item[k]*factor
            
        '''
        After we have assigned to each leaf their possibilities of each labels, we are using the training dataset
        to find out what is the product of their possibilities of beeing classified correctly
        '''
        product_of_leafs_probabilities  = []
        k=0 
        for datum in X_train:
            current_node = 0
            #make sure that we are not in leafs
            while current_node not in leafs:
    
                if  datum[features[current_node]] > thresholds[current_node]:
                    current_node = (current_node*2)+2
                else:
                    current_node = (current_node*2)+1
                
                if current_node in leafs:
                    indice = leafs.index(current_node)#find the position of the dictionary probabilities given the leaf number
                    probs = leafs_possibilities[indice]
                    for prob in probs:
                        target_probability = probs[y_train[k]]
                        '''
                        we make sure that in the case we are on a homogeneous leaf, 
                        we dont get a 0 probability but a very low one
                        '''
                        
                        if target_probability == 0:
                            target_probability = 0.01
                        if target_probability == 1:
                            target_probability = 0.98
                            
                    product_of_leafs_probabilities.append((target_probability))
            k+=1

        product_of_target_feature = np.prod(product_of_leafs_probabilities)
        return product_of_target_feature, leafs_possibilities
    
    #(theta|T)
    def features_and_threshold_probabilities(X_train,features,thresholds):
        #this need to change
        probabilities = []
                
        '''
            the likelihood of choosing the specific feature and threshold must be computed. We need to find out the probabilty 
            of selecting the specific feature multiplied by 1/the margins. it should be 
            (1/number of features) * (1/(upper bound-lower bound)) 
        '''
        for i in range(len(features)):
            probabilities.append( 1/len(X_train[0]) * 1/ (max(X_train[:,features[i]]) - min(X_train[:,features[i]])) )
            #probabilities.append(math.log( 1/len(X_train[0]) * 1/len(X_train[:]) ))
            
        probability = np.prod(probabilities)
        return ((probability))
    
    #p(T)
    def prior_calculation(leafs,a,b):
        depth = math.ceil(math.log(len(leafs),2))
        prior = a / ((1+depth)**b)    
        print("prior:", prior)
        print("depth:", depth)
        return (prior)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
