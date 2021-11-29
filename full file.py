# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:07:27 2021

@author: efthi
"""
import random 
import collections
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import math
import copy
import pandas as pd
# data = datasets.load_wine()
# X = data.data
# y = data.target

data = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\Reduced features space decision tree\Dataset2\winequality-red.csv" )
X = data.drop(["Target"], axis = 1)
y = data.Target


X = X.to_numpy()
y = y.to_numpy()




X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=5)


class Tree():

    def initialise_tree(X_train):
        leafs = [1,2]
       
        feature = random.randint(0,len(X_train[0])-1)
        threshold = random.randint(0,len(X_train)-1)
        tree = [[0, 1, 2, feature, X_train[threshold,feature]]]
        return tree, leafs
    
    
    def grow(X_train, tree, leafs):
            
            action = "grow"
            print(action)
            '''
            grow tree by just simply creating the individual nodes. each node holds their node index, 
            the left and right leaf index, the node feature and threshold
            '''
            random_index = random.randint(0,len(leafs)-1)
            leaf_to_grow = leafs[random_index] 
    
            #generating a random faeture
            feature = random.randint(0,len(X_train[0])-1)
            #generating a random threshold
            threshold = random.randint(0,len(X_train)-1)
            threshold = (X_train[threshold,feature])
            
            node = [leaf_to_grow, max(leafs)+1, max(leafs)+2, feature, threshold]
    
            #add the new leafs on the leafs array
            leafs.append(max(leafs)+1)
            leafs.append(max(leafs)+1)
            #delete from leafs the new node
            leafs.remove(leaf_to_grow)
            tree.append(node)

            
            return tree, leafs, action
        
        
        
    def prune(tree, leafs):
        action = "prune"
        print(action)

        '''
        For example when we have nodes 0,1,2 and leafs 3,4,5,6 when we prune we take the leafs 6 and 5 out, and the
        node 2, now becomes a leaf. 
        '''
        random_index = random.randint(0,len(tree)-1)
        node_to_prune = tree[random_index]
        print("node to prune: ", node_to_prune)
        while random_index == 0 : 
            random_index = random.randint(0,len(tree)-1)
            node_to_prune = tree[random_index]
            print(random_index)
            print("node to prune: ", node_to_prune)


        if (node_to_prune[1] in leafs) and (node_to_prune[2] in leafs):
            
            #remove the pruned leafs from leafs list and add the node as a leaf
            leafs.append(node_to_prune[0])
            leafs.remove(node_to_prune[1])
            leafs.remove(node_to_prune[2])
            #delete the specific node from the node lists
            del tree[random_index]
        else:
            
            delete_node_indices = []
            i=0
            for node in tree:
                if node_to_prune[1] == node[0] or node_to_prune[2] == node[0]:
                    delete_node_indices.append(node)
                i+=1
            tree.remove(node_to_prune)
            for node in delete_node_indices:
                 tree.remove(node)
            
            for i in range(len(tree)):
                for p in range(1,len(tree)):
                    count = 0
                    for k in range(len(tree)-1):
                        if tree[p][0] == tree[k][1] or tree[p][0] == tree[k][2]:
                            count = 1
                    if count == 0:
                        tree.remove(tree[p])
                        break
                    

        print("tree after prune: ", tree)
        new_leafs = []
        for node in tree:
            count1 = 0
            count2 = 0
            for check_node in tree:
                if node[1] == check_node[0]:
                    count1 = 1
                if node[2] == check_node[0]:
                    count2 = 1
                
            if count1 == 0:
                new_leafs.append(node[1])
                
            if count2 == 0:
                new_leafs.append(node[2])
            
           
        leafs[:] = new_leafs[:] 
        return tree, leafs, action
        
        
    def change(X_train, tree):
        action = "change"
        print(action)
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have chosen 
        and then pick unoformly a node and change their features and thresholds
        '''
        random_index = random.randint(0,len(tree)-1)
        node_to_change = tree[random_index]
        new_feature = random.randint(0,len(X_train[0])-1)
        new_threshold = random.randint(0,len(X_train)-1)
        node_to_change[3] = new_feature
        node_to_change[4] = X_train[new_threshold, new_feature]   
         
        return tree,action
    
    def swap(tree):
            action = "swap"
            print(action)
            '''
            need to swap the features and the threshold among the 2 nodes
            '''
            random_index_1 = random.randint(0,len(tree)-1)
            random_index_2 = random.randint(0,len(tree)-1)
            node_to_swap1 = tree[random_index_1]
            node_to_swap2 = tree[random_index_2]

            #in case we choose the same node
            while node_to_swap1 == node_to_swap2:
                random_index_2 = random.randint(0,len(tree)-1)
                node_to_swap2 = tree[random_index_2]

            temporary_feature = node_to_swap1[3]
            temporary_threshold = node_to_swap1[4]

            node_to_swap1[3]= node_to_swap2[3]
            node_to_swap1[4] = node_to_swap2[4]
    
            node_to_swap2[3] = temporary_feature
            node_to_swap2[4] = temporary_threshold
            
            return tree, action

class test():
    #Π(Y_i|T,theta,x_i)
    def calculate_leaf_occurences(X_train, y_train, tree, leafs):
        '''
        we calculate how many labelled as 0 each leaf has, how many labelled as 1 each leaf has and so on
        '''
        leaf_occurences = []
        k=0 
        for leaf in leafs:
            leaf_occurences.append([leaf])
            
        for datum in X_train:
            flag = "false"
            current_node = tree[0]
            
            #make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in leafs and flag == "false":
                if  datum[current_node[3]] > current_node[4]:
                    for node in tree:
                        if node[0] == current_node[2]:
                            current_node = node
                            break
                        if current_node[2] in leafs:
                            leaf = current_node[2]
                            flag = "true"
                            break
                            
                else:
                    for node in tree:
                        if node[0] == current_node[1]:
                            current_node = node
                            break
                        if current_node[1] in leafs:
                            leaf = current_node[1]
                            flag = "true"
                            break
    
            '''
            create a list of lists that holds the leafs and the number of occurences
            for example [[4,1,1,2,2,2][5,1,1,2,2,1][6,1,2,2,1,2][7,2,2,2,1,2,1,2]]
            The first number represents the leaf id number
            '''

            
                
            for item in leaf_occurences:

                if item[0] == leaf:
                    item.append(y_train[k])
            k+=1

        '''
        we have some cases where some leaf nodes may do not have any probabilities
        because no data point ended up in the particular leaf
        We add equal probabilities for each label to the particular leaf.
        For example if we have 4 labels, we add 0:0.25, 1:0.25, 2:0.25, 3:0.25
        '''
        
        for item in leaf_occurences:
            if len(item) == 1 :
                unique = set(y_train)
                unique = list(unique)
                for i in range(len(unique)):
                    item.append(i)
                

        leaf_occurences = sorted(leaf_occurences)
        leafs = sorted(leafs)

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
        

        product_of_leafs_probabilities  = []
        k=0     
        for datum in X_train:
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            flag = "false"
            current_node = tree[0]
            #make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in leafs and flag == "false":
                if  datum[current_node[3]] > current_node[4]:
                    for node in tree:
                        if node[0] == current_node[2]:
                            current_node = node
                            break
                        if current_node[2] in leafs:
                            leaf = current_node[2]
                            #print(leaf)
                            flag = "true"
                            break
                            
                else:
                    for node in tree:
                        if node[0] == current_node[1]:
                            current_node = node
                            break
                        if current_node[1] in leafs:
                            leaf = current_node[1]
                            #print(leaf)
                            flag = "true"
                            break
                        
            if leaf in leafs:
                indice = leafs.index(leaf)#find the position of the dictionary probabilities given the leaf number
                probs = leafs_possibilities[indice]
                for prob in probs:
                    target_probability = probs[y_train[k]]


                    '''
                    we make sure that in the case we are on a homogeneous leaf, 
                    we dont get a 0 probability but a very low one
                    '''
                    
                    if target_probability == 0:
                        target_probability = 0.02
                    if target_probability == 1:
                        target_probability = 0.98
                
                product_of_leafs_probabilities.append(math.log(target_probability))


            k+=1
        product_of_target_feature = np.sum(product_of_leafs_probabilities)
        return product_of_target_feature, leafs_possibilities
    
    #(theta|T)
    def features_and_threshold_probabilities(X_train, tree):
        #this need to change
        probabilities = []
                
        '''
            the likelihood of choosing the specific feature and threshold must be computed. We need to find out the probabilty 
            of selecting the specific feature multiplied by 1/the margins. it should be 
            (1/number of features) * (1/(upper bound-lower bound)) 
        '''
        for node in tree:
            probabilities.append( 1/len(X_train[0]) * 1/ (max(X_train[:,node[3]]) - min(X_train[:,node[3]])) )
            #probabilities.append(math.log( 1/len(X_train[0]) * 1/len(X_train[:]) ))
        
        probability = np.prod(probabilities)
        return ((probability))
    
    
    #p(T)
    def prior_calculation(tree,a,b):
        i = len(tree) - 1
        depth = 0
        while i >= 0 :
            node = tree[i]
            next_node = tree[i-1]
            if node[0] == next_node[1]:
                depth+=1
            if node[0] == next_node[2]:
                depth +=1
            i -= 1
        depth = depth + 1
        prior = a / ((1+depth)**b)    
        return (prior)
        
class TreeDistribution():    
    
    def sample(X_train, tree, leafs):

        
        #initialise the probabilities of each move
        moves = ["prune", "swap", "change", "grow"]
        moves_prob = [0.5, 0.2, 0.2, 0.1]
        moves_probabilities = np.cumsum(moves_prob)
        random_number= random.random()
        if random_number < moves_probabilities [0] and len(tree) > 1:
            #prune
            #(T',theta'|T,theta,a)
            #print("prune")
            forward_probability = moves_prob[0] *( 1/(len(tree)-1))

            Tree.prune(tree, leafs)
            

            #(T,theta|T',theta',a)
            reverse_probability = moves_prob[3] * (1/len(X_train[0])) * (1/len(X_train[:])) * (1 / len(leafs))
            
    
        elif random_number < moves_probabilities[1] and len(tree) > 1:
            #swap
            #(T',theta'|T,theta,a)
            forward_probability = moves_prob[1] * 1/ ((len(tree) * (len(tree) -1))/2)
            Tree.swap(tree)
            #(T,theta|T',theta',a)
            reverse_probability = moves_prob [1] * 1/((len(tree) * (len(tree) -1))/2)
            
            
        
        elif random_number < moves_probabilities[2]:
            #change
            #(T',theta'|T,theta,a)
            forward_probability = moves_prob [2] * (1/len(tree)) * 1/len(X_train[0]) * 1/len(X_train[:])
            Tree.change(X_train, tree)
            #(T,theta|T',theta',a)
            reverse_probability = moves_prob [2] * (1/len(tree)) * 1/len(X_train[0]) * 1/len(X_train[:])
            
        
        else:
            #grow
            #(T',theta'|T,theta,a)
            forward_probability = moves_prob[3] * (1/len(X_train[0])) * (1/len(X_train[:])) * (1 / len(leafs))
            Tree.grow(X_train, tree, leafs)
            #(T,theta|T',theta',a)
            reverse_probability = moves_prob[0] * (1/(len(tree)-1))
        

        return (forward_probability), (reverse_probability)
        
        
    def eval(X_train, y_train, tree, leafs, a, b):
        #call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = test.calculate_leaf_occurences(X_train, y_train, tree, leafs)
        target1 = (target1)
        #call test tree to calculate  (theta|T)
        target2 = test.features_and_threshold_probabilities(X_train, tree)
        target2 = math.log(target2)
        #p(T)
        target3 = test.prior_calculation(tree, a, b)
        target3 = math.log(target3)
        # print("target1: ", target1)
        # print("target2: ", target2)
        # print("target3: ", target3)
        return (target1+target2+target3) , leafs_possibilities_for_prediction    


def predict (X_test, tree,  leafs, leaf_possibilities):
    leafs = sorted(leafs)
    labels = []
    for datum in X_test:
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        flag = "false"
        current_node = tree[0]
        label_max = -1
        #make sure that we are not in leafs. current_node[0] is the node
        while current_node[0] not in leafs and flag == "false":
            if  datum[current_node[3]] > current_node[4]:
                for node in tree:
                    if node[0] == current_node[2]:
                        current_node = node
                        break
                    if current_node[2] in leafs:
                        leaf = current_node[2]
                        flag = "true"
                        indice = leafs.index(leaf)
                        for x,y in leaf_possibilities[indice].items():
                            if y == label_max :
                                actual_label = 1
                        
                            if y > label_max:
                                label_max = y
                                actual_label = x
                                                
  
                        labels.append(actual_label)
                        break
                        
            else:
                for node in tree:
                    if node[0] == current_node[1]:
                        current_node = node
                        break
                    if current_node[1] in leafs:
                        leaf = current_node[1]
                        flag = "true"
                        indice = leafs.index(leaf)
                        for x,y in leaf_possibilities[indice].items():
                            if y == label_max :
                                actual_label = 1
                        
                            if y > label_max:
                                label_max = y
                                actual_label = x
                                                
  
                        labels.append(actual_label)
                        break


  

        if current_node in leafs:
            indice = current_node.index(current_node)
            #find in the dictionary which is the highest probable label
            for x,y in leaf_possibilities[indice].items():
                if y == label_max :
                    actual_label = 1
                    
                if y > label_max:
                    label_max = y
                    actual_label = x
    return (labels)
    
def accuracy(y_test, labels):
    correct_classification = 0
    for i in range(len(y_test)):
        if labels[i] == y_test[i]:
            correct_classification +=1
    
    acc = correct_classification*100/len(y_test)
    return acc
    correct_classification = 0
    for i in range(len(y_test)):
        if labels[i] == y_test[i]:
            correct_classification +=1
    
    acc = correct_classification*100/len(y_test)
    return acc


sampleTree, sampleLeafs = Tree.initialise_tree(X_train)
currentTree = copy.deepcopy(sampleTree)
currentLeafs = copy.deepcopy(sampleLeafs)

a = 0.1
b = 5

forward_prob = []
accepted = 0
state = "rejected"
forest = []
for i in range (10000):
    print("")
    print("")
    print("")
    
    #the bolow command can run outside for loop. current_tree_target is the same if rejected, otherwise is updated to new_tree_target
    current_tree_target, predict_possibilities_current = TreeDistribution.eval(X_train, y_train, sampleTree, sampleLeafs, a, b)
    
    # print("current tree target: ", current_tree_target)
    forward_probability, reverse_probability = TreeDistribution.sample(X_train, sampleTree, sampleLeafs)
    
    forward_prob.append(forward_probability)
    forward_probability = np.sum(forward_prob)
    reverse_probability = forward_probability + reverse_probability
    print("old_tree: ", currentTree)
    print("")
    print("new_tree: ", sampleTree)
    new_tree_target, predict_possibilities_new = TreeDistribution.eval(X_train, y_train, sampleTree, sampleLeafs, a, b)
    
    targetRatio = (new_tree_target) - (current_tree_target)    
    proposalRatio = math.log(reverse_probability) - math.log(forward_probability)
    if targetRatio <709:
        acceptProbability = min(1,  math.exp(targetRatio + proposalRatio))
    else: 
        acceptProbability = 1
    # print("new tree target: ", new_tree_target)
    # print("targetRatio: ", targetRatio)
    # print("ProposalRatio: ", proposalRatio)
    # print("acceptProbability: ", acceptProbability)

    q= random.random()
    if (acceptProbability > q):
        currentTree = copy.deepcopy(sampleTree)
        currentLeafs = copy.deepcopy(sampleLeafs)
        predict_possibilities_current = predict_possibilities_new
        state = "accepted"
        print("new tree accepted")
        accepted +=1
    else:
        sampleTree = copy.deepcopy(currentTree)
        sampleLeafs = copy.deepcopy(currentLeafs)
        del forward_prob[-1]
        state = "rejected"
        print("new tree rejected")
    
            
    # if i >12000 and state == "accepted" :
        
    #     labels = predict (X_test, currentTree,  currentLeafs, predict_possibilities_current)
    #     forest.append(labels)


        

labels = predict (X_test, currentTree,  currentLeafs, predict_possibilities_current)



    
# predictions = pd.DataFrame(forest)
# for column in predictions:
#     labels.append(predictions[column].mode())
    
acc = accuracy(y_test, labels)   
            
            
