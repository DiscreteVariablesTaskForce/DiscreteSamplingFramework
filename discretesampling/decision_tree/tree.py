import random
import math
import numpy as np
import collections

class Tree():
    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        tree, leafs = self.initialise_tree()
        self.tree = tree
        self.leafs = leafs
        self.lastAction = ""
        
    def initialise_tree(self):
        leafs = [1,2]
       
        feature = random.randint(0,len(self.X_train[0])-1)
        threshold = random.randint(0,len(self.X_train)-1)
        tree = [[0, 1, 2, feature, self.X_train[threshold,feature]]]
        return tree, leafs
    
    
    def grow(self):
            
            action = "grow"
            print(action)
            self.lastAction = action
            '''
            grow tree by just simply creating the individual nodes. each node holds their node index, 
            the left and right leaf index, the node feature and threshold
            '''
            random_index = random.randint(0,len(self.leafs)-1)
            leaf_to_grow = self.leafs[random_index] 
    
            #generating a random faeture
            feature = random.randint(0,len(self.X_train[0])-1)
            #generating a random threshold
            threshold = random.randint(0,len(self.X_train)-1)
            threshold = (self.X_train[threshold,feature])
            
            node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature, threshold]
    
            #add the new leafs on the leafs array
            self.leafs.append(max(self.leafs)+1)
            self.leafs.append(max(self.leafs)+1)
            #delete from leafs the new node
            self.leafs.remove(leaf_to_grow)
            self.tree.append(node)
            

            
            return self
        
        
        
    def prune(self):
        action = "prune"
        print(action)
        self.lastAction = action
        '''
        For example when we have nodes 0,1,2 and leafs 3,4,5,6 when we prune we take the leafs 6 and 5 out, and the
        node 2, now becomes a leaf. 
        '''
        random_index = random.randint(0,len(self.tree)-1)
        node_to_prune = self.tree[random_index]
        while random_index == 0 : 
            random_index = random.randint(0,len(self.tree)-1)
            node_to_prune = self.tree[random_index]


        if (node_to_prune[1] in self.leafs) and (node_to_prune[2] in self.leafs):
            
            #remove the pruned leafs from leafs list and add the node as a leaf
            self.leafs.append(node_to_prune[0])
            self.leafs.remove(node_to_prune[1])
            self.leafs.remove(node_to_prune[2])
            #delete the specific node from the node lists
            del self.tree[random_index]
        else:
            
            delete_node_indices = []
            i=0
            for node in self.tree:
                if node_to_prune[1] == node[0] or node_to_prune[2] == node[0]:
                    delete_node_indices.append(node)
                i+=1
            self.tree.remove(node_to_prune)
            for node in delete_node_indices:
                 self.tree.remove(node)
            
            for i in range(len(self.tree)):
                for p in range(1,len(self.tree)):
                    count = 0
                    for k in range(len(self.tree)-1):
                        if self.tree[p][0] == self.tree[k][1] or self.tree[p][0] == self.tree[k][2]:
                            count = 1
                    if count == 0:
                        self.tree.remove(self.tree[p])
                        break
                    

        new_leafs = []
        for node in self.tree:
            count1 = 0
            count2 = 0
            for check_node in self.tree:
                if node[1] == check_node[0]:
                    count1 = 1
                if node[2] == check_node[0]:
                    count2 = 1
                
            if count1 == 0:
                new_leafs.append(node[1])
                
            if count2 == 0:
                new_leafs.append(node[2])
            
           
        self.leafs[:] = new_leafs[:] 
        return self
        
    def change(self):
        action = "change"
        print(action)
        self.lastAction = action
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have chosen 
        and then pick unoformly a node and change their features and thresholds
        '''
        random_index = random.randint(0,len(self.tree)-1)
        node_to_change = self.tree[random_index]
        new_feature = random.randint(0,len(self.X_train[0])-1)
        new_threshold = random.randint(0,len(self.X_train)-1)
        node_to_change[3] = new_feature
        node_to_change[4] = self.X_train[new_threshold, new_feature]   
         
        return self
    
    def swap(self):
            action = "swap"
            print(action)
            self.lastAction = action
            '''
            need to swap the features and the threshold among the 2 nodes
            '''
            random_index_1 = random.randint(0,len(self.tree)-1)
            random_index_2 = random.randint(0,len(self.tree)-1)
            node_to_swap1 = self.tree[random_index_1]
            node_to_swap2 = self.tree[random_index_2]

            #in case we choose the same node
            while node_to_swap1 == node_to_swap2:
                random_index_2 = random.randint(0,len(self.tree)-1)
                node_to_swap2 = self.tree[random_index_2]

            temporary_feature = node_to_swap1[3]
            temporary_threshold = node_to_swap1[4]

            node_to_swap1[3]= node_to_swap2[3]
            node_to_swap1[4] = node_to_swap2[4]
    
            node_to_swap2[3] = temporary_feature
            node_to_swap2[4] = temporary_threshold
            
            return self
    
    #Π(Y_i|T,theta,x_i)
    def calculate_leaf_occurences(self):
        '''
        we calculate how many labelled as 0 each leaf has, how many labelled as 1 each leaf has and so on
        '''
        leaf_occurences = []
        k=0 
        for leaf in self.leafs:
            leaf_occurences.append([leaf])
            
        for datum in self.X_train:
            flag = "false"
            current_node = self.tree[0]
            
            #make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in self.leafs and flag == "false":
                if  datum[current_node[3]] > current_node[4]:
                    for node in self.tree:
                        if node[0] == current_node[2]:
                            current_node = node
                            break
                        if current_node[2] in self.leafs:
                            leaf = current_node[2]
                            flag = "true"
                            break
                            
                else:
                    for node in self.tree:
                        if node[0] == current_node[1]:
                            current_node = node
                            break
                        if current_node[1] in self.leafs:
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
                    item.append(self.y_train[k])
            k+=1

        '''
        we have some cases where some leaf nodes may do not have any probabilities
        because no data point ended up in the particular leaf
        We add equal probabilities for each label to the particular leaf.
        For example if we have 4 labels, we add 0:0.25, 1:0.25, 2:0.25, 3:0.25
        '''
        
        for item in leaf_occurences:
            if len(item) == 1 :
                unique = set(self.y_train)
                unique = list(unique)
                for i in range(len(unique)):
                    item.append(i)
                

        leaf_occurences = sorted(leaf_occurences)
        leafs = sorted(self.leafs)

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
        for datum in self.X_train:
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            flag = "false"
            current_node = self.tree[0]
            #make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in leafs and flag == "false":
                if  datum[current_node[3]] > current_node[4]:
                    for node in self.tree:
                        if node[0] == current_node[2]:
                            current_node = node
                            break
                        if current_node[2] in leafs:
                            leaf = current_node[2]
                            #print(leaf)
                            flag = "true"
                            break
                            
                else:
                    for node in self.tree:
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
                    target_probability = probs[self.y_train[k]]


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
    def features_and_threshold_probabilities(self):
        #this need to change
        probabilities = []
                
        '''
            the likelihood of choosing the specific feature and threshold must be computed. We need to find out the probabilty 
            of selecting the specific feature multiplied by 1/the margins. it should be 
            (1/number of features) * (1/(upper bound-lower bound)) 
        '''
        for node in self.tree:
            probabilities.append( 1/len(self.X_train[0]) * 1/ (max(self.X_train[:,node[3]]) - min(self.X_train[:,node[3]])) )
            #probabilities.append(math.log( 1/len(X_train[0]) * 1/len(X_train[:]) ))
        
        probability = np.prod(probabilities)
        return ((probability))
    
    def evaluatePrior(self,a,b):
        i = len(self.tree) - 1
        depth = 0
        while i >= 0 :
            node = self.tree[i]
            next_node = self.tree[i-1]
            if node[0] == next_node[1]:
                depth+=1
            if node[0] == next_node[2]:
                depth +=1
            i -= 1
        depth = depth + 1
        prior = a / ((1+depth)**b)    
        return (prior)
        
    def getLeafPossibilities(self):
        target1, leafs_possibilities_for_prediction = self.calculate_leaf_occurences()
        return leafs_possibilities_for_prediction
    
    def evaluatePosterior(self, a, b):
        #call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = self.calculate_leaf_occurences()
        #call test tree to calculate  (theta|T)
        target2 = self.features_and_threshold_probabilities()
        target2 = math.log(target2)
        #p(T)
        target3 = self.evaluatePrior(a, b)
        target3 = math.log(target3)

        return (target1+target2+target3) 
    
        
        