import math
import collections
import numpy as np
from .. import discrete

class TreeTarget(discrete.DiscreteVariableTarget):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def eval(self, x):
        #call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = self.calculate_leaf_occurences(x)
        #call test tree to calculate  (theta|T)
        target2 = self.features_and_threshold_probabilities(x)
        target2 = math.log(target2)
        #p(T)
        target3 = self.evaluatePrior(x)
        target3 = math.log(target3)
        return (target1+target2+target3)
        
    #Π(Y_i|T,theta,x_i)
    def calculate_leaf_occurences(self,x):
        '''
        we calculate how many labelled as 0 each leaf has, how many labelled as 1 each leaf has and so on
        '''
        leaf_occurences = []
        k=0 
        for leaf in x.leafs:
            leaf_occurences.append([leaf])
            
        for datum in x.X_train:
            flag = "false"
            current_node = x.tree[0]
            
            #make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in x.leafs and flag == "false":
                if  datum[current_node[3]] > current_node[4]:
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
            create a list of lists that holds the leafs and the number of occurences
            for example [[4,1,1,2,2,2][5,1,1,2,2,1][6,1,2,2,1,2][7,2,2,2,1,2,1,2]]
            The first number represents the leaf id number
            '''

            
                
            for item in leaf_occurences:

                if item[0] == leaf:
                    item.append(x.y_train[k])
            k+=1

        '''
        we have some cases where some leaf nodes may do not have any probabilities
        because no data point ended up in the particular leaf
        We add equal probabilities for each label to the particular leaf.
        For example if we have 4 labels, we add 0:0.25, 1:0.25, 2:0.25, 3:0.25
        '''
        
        for item in leaf_occurences:
            if len(item) == 1 :
                unique = set(x.y_train)
                unique = list(unique)
                for i in range(len(unique)):
                    item.append(i)
                

        leaf_occurences = sorted(leaf_occurences)
        leafs = sorted(x.leafs)

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
        for datum in x.X_train:
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            flag = "false"
            current_node = x.tree[0]
            #make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in leafs and flag == "false":
                if  datum[current_node[3]] > current_node[4]:
                    for node in x.tree:
                        if node[0] == current_node[2]:
                            current_node = node
                            break
                        if current_node[2] in leafs:
                            leaf = current_node[2]
                            #print(leaf)
                            flag = "true"
                            break
                            
                else:
                    for node in x.tree:
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
                    target_probability = probs[x.y_train[k]]


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
    def features_and_threshold_probabilities(self, x):
        #this need to change
        probabilities = []
                
        '''
            the likelihood of choosing the specific feature and threshold must be computed. We need to find out the probabilty 
            of selecting the specific feature multiplied by 1/the margins. it should be 
            (1/number of features) * (1/(upper bound-lower bound)) 
        '''
        for node in x.tree:
            probabilities.append( 1/len(x.X_train[0]) * 1/ (max(x.X_train[:,node[3]]) - min(x.X_train[:,node[3]])) )
            #probabilities.append(math.log( 1/len(X_train[0]) * 1/len(X_train[:]) ))
        
        probability = np.prod(probabilities)
        return ((probability))
    
    def evaluatePrior(self,x):
        i = len(x.tree) - 1
        depth = 0
        while i >= 0 :
            node = x.tree[i]
            next_node = x.tree[i-1]
            if node[0] == next_node[1]:
                depth+=1
            if node[0] == next_node[2]:
                depth +=1
            i -= 1
        depth = depth + 1
        prior = self.a / ((1+depth)**self.b)    
        return (prior)
    
    def getLeafPossibilities(self, x):
        target1, leafs_possibilities_for_prediction = self.calculate_leaf_occurences(x)
        return leafs_possibilities_for_prediction