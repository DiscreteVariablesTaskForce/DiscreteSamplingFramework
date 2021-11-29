import collections
import numpy as np
import math
import copy

class test():
    
    def __init__(self, X_train, y_train, nodes, leafs):
       super.__init__()
       self.X_train
       self.y_train
       self.nodes
       self.leafs
    
    
    
    #p(T)
    def prior_calculation(self,a,b):
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
    
    
