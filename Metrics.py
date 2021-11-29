from create_tree import Tree


class stats():
    def __init__(self, tree, X_test):
        self.X_train = tree.X_train
        self.y_train = tree.y_train
        self.tree = tree.nodes
        self.leafs = tree.leafs
        
        
    def predict (self, X_test):
        leaf_possibilities = self.getLeafPossibilities()
        leafs = sorted(self.leafs)
        labels = []
        for datum in X_test:
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            flag = "false"
            current_node = self.tree[0]
            label_max = -1
            #make sure that we are not in leafs. current_node[0] is the node
            while current_node[0] not in leafs and flag == "false":
                if  datum[current_node[3]] > current_node[4]:
                    for node in self.tree:
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
                    for node in self.tree:
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
