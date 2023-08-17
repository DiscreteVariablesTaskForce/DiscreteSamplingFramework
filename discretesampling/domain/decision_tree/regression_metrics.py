# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:29:10 2023

@author: efthi
"""

import math
import numpy as np
import collections
from scipy.stats import norm
from discretesampling.domain.decision_tree.metrics import stats


class RegressionStats(stats):
    
    def regression_predict(self, X_test):
        leaf_possibilities = getLeafPossibilities(self.tree)
        leafs = sorted(self.tree.leafs)
        labels = []
        for datum in X_test:
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            flag = "false"
            current_node = self.tree.tree[0]
            #label_max = -1
            # make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in leafs and flag == "false":
                if datum[current_node[3]] > current_node[4]:
                    
                    for node in self.tree.tree:
                        if node[0] == current_node[2]:
                            current_node = node
                            break
                        if current_node[2] in leafs:
                            leaf = current_node[2]
                            flag = "true"
                            indice = leafs.index(leaf)
                            #print(leaf_possibilities[indice])
                            labels.append(leaf_possibilities[indice])
                            break

                else:
                    for node in self.tree.tree:
                        if node[0] == current_node[1]:
                            current_node = node
                            break
                        if current_node[1] in leafs:
                            leaf = current_node[1]
                            flag = "true"
                            indice = leafs.index(leaf)
                            #print(leaf_possibilities[indice])
                            labels.append(leaf_possibilities[indice])
                            break

           
        return (labels)
    
def accuracy_mse(y_test, labels):
    squared_error = []
    #print("accuracy calculation starts:")
    for i in range(len(y_test)):
        #print("this: ", y_test[i], "with: ", labels[i])
        squared_error.append((y_test[i]-labels[i])**2)
        
    
    #print(np.sum(squared_error)/len(y_test))
    return (np.sum(squared_error)/len(y_test))

def getLeafPossibilities(x):
    target1, leafs_possibilities_for_prediction = regression_likelihood(x)
    
    return leafs_possibilities_for_prediction


# Π(Y_i|T,theta,x_i)
def regression_likelihood(x):
    
 
    '''
    we calculate how many labelled as 0 each leaf has, how many labelled as 1
    each leaf has and so on
    '''
    
 
    leaf_occurences = []
    k = 0
    for leaf in x.leafs:
        leaf_occurences.append([leaf])

    for datum in x.X_train:
        flag = "false"
        current_node = x.tree[0]

        # make sure that we are not in leafs. current_node[0] is the node
        while current_node[0] not in x.leafs and flag == "false":
            #print("compare this: ", datum[current_node[3]], "with this: ", current_node[4])
            if datum[current_node[3]] > current_node[4]:
                for node in x.tree:
                    if node[0] == current_node[2]:
                        current_node = node
                        break
                    if current_node[2] in x.leafs:
                        leaf = current_node[2]
                        flag = "true"
                        break

            else:
                for node in x.tree:
                    if node[0] == current_node[1]:
                        current_node = node
                        break
                    if current_node[1] in x.leafs:
                        leaf = current_node[1]
                        flag = "true"
                        break

        '''
        create a list of lists that holds the leafs and the number of
        occurences for example [[4, 1, 1, 2, 2, 2], [5, 1, 1, 2, 2, 1],
        [6, 1, 2, 2, 1, 2], [7, 2, 2, 2, 1, 2, 1, 2]]
        The first number represents the leaf id number
        '''
        
        
        for item in leaf_occurences:
            if item[0] == leaf:
                item.append(x.y_train[k])
        k += 1
       
    
       
        
       
    '''
    we have some cases where some leaf nodes when it is clssification may do not have any probabilities
    because no data point ended up in the particular leaf
    We add equal probabilities for each label to the particular leaf.
    For example if we have 4 labels, we add 0:0.25, 1:0.25, 2:0.25, 3:0.25
    when it comes to regression we just add the mean value y_train
    '''
    
        
    ''' This is when it comes to classification
    for item in leaf_occurences:
        if len(item) == 1:
            unique = set(x.y_train)
            unique = list(unique)
            for i in range(len(unique)):
                item.append(i)
    '''
    penalty = 0
    for item in leaf_occurences:
        if len(item) == 1:
            penalty+=1
            item.append(np.mean(x.y_train))
            #item.append(1000000000000)
    

    leaf_occurences = sorted(leaf_occurences)
    leafs = sorted(x.leafs)
    
    
    
    '''
    we then delete the first number of the list which represents the leaf node
    id.
    '''
    
    
    for i in range(len(leaf_occurences)):
        new_list = True
        for p in range(len(leaf_occurences[i])):
            if new_list:
                new_list = False
                del leaf_occurences[i][p]
    
    #print("leaf occurences: ", leaf_occurences)
    '''
    first count the number of labels in each leaf.
    Then create probabilities by normalising those values[0,1]
    '''
    
    leaf_values = []
    for leaf in leaf_occurences:
        leaf_values.append(np.mean(leaf))
    
    
    '''
    # not usefull for regression
    leafs_possibilities = []
    for number_of_leaves in range(len(leaf_occurences)):
        occurrences = collections.Counter(leaf_occurences[number_of_leaves][:])
        leafs_possibilities.append(occurrences)
    
    
    # create leafs possibilities/not useful for regression
    for item in leafs_possibilities:
        factor = 1.0/sum(item.values())
        for k in item:
            item[k] = item[k]*factor
    
    '''
    product_of_leafs_probabilities = []
    squared_error = []
    predicted = []
    k = 0
    for datum in x.X_train:
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        flag = "false"
        current_node = x.tree[0]
        # make sure that we are not in leafs. current_node[0] is the node
        while current_node[0] not in leafs and flag == "false":
            if datum[current_node[3]] > current_node[4]:
                for node in x.tree:
                    if node[0] == current_node[2]:
                        current_node = node
                        break
                    if current_node[2] in leafs:
                        leaf = current_node[2]
                        # print(leaf)
                        flag = "true"
                        break

            else:
                for node in x.tree:
                    if node[0] == current_node[1]:
                        current_node = node
                        break
                    if current_node[1] in leafs:
                        leaf = current_node[1]
                        # print(leaf)
                        flag = "true"
                        break

        if leaf in leafs:
            # find the position of the dictionary probabilities given the leaf
            indice = leafs.index(leaf)        
            probs = leaf_values[indice]
            predicted.append(probs)
            squared_error.append((probs-x.y_train[k])**2)#caclulate the squared error
           
        k += 1
    
    log_likelihood = -(len(x.y_train)/2) * (-np.log(2)+np.log(sum(squared_error)))
    
    print("likelihood: ", sum(squared_error)/len(x.y_train))
    return log_likelihood, leaf_values #math.log(likelihood), leaf_values
    