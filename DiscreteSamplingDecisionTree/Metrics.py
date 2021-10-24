# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 10:30:36 2021

@author: efthi
"""

def predict(X_test, old_tree, leaf_possibilities):
    '''
    gets the old_tree parameters and makes predictions on the test dataset based on the train dataset
    '''
    labels = []
    p = 0
    for datum in X_test:
        current_node = 0
        label_max = -1 #helper variable to keep the highest variable
        while current_node not in old_tree[1]:
            if  datum[old_tree[2][current_node]] > old_tree[3][current_node]:
                current_node = (current_node*2)+2
            else:
                current_node = (current_node*2)+1
            p+=1
            if current_node in old_tree[1]:
                indice = old_tree[1].index(current_node)
                #find in the dictionary which is the highest probable label
                for x,y in leaf_possibilities[indice].items():
                    if y == label_max :
                        actual_label = 1
                        
                    if y > label_max:
                        label_max = y
                        actual_label = x
                                                
  
                labels.append(actual_label)
    return labels

def accuracy(y_test, labels):
    correct_classification = 0
    print("labels: ", len(labels))
    print("y_test: ", len(y_test))
    for i in range(len(y_test)):
        if labels[i] == y_test[i]:
            correct_classification +=1
    
    acc = correct_classification*100/len(y_test)
    return acc