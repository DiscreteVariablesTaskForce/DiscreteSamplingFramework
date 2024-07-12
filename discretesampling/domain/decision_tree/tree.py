import copy
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base import types
from discretesampling.domain.decision_tree.util import encode_move, decode_move, extract_tree, extract_leafs
from discretesampling.domain.decision_tree.tree_distribution import TreeProposal
from discretesampling.domain.decision_tree.tree_target import TreeTarget
from math import log, ceil


# MJAS notes as of 2024/06/28 on the code [before improvements]:
# this code is not properly documented... should be a rationale for the class design - e.g. what are features and labels for when we are passing in X_train and Y_train too? Even the likelihood function uses it's own separate inputs
# this is not good class design... SUGGESTIONS:
# - would make sense to use a dictionary, or a pandas table row, or a new class to represent nodes (not a mixed-type list)
# - also it should not be necessary to search for node ids using while loops... use a keyed data stucture like a pandas table or keep an index
# - maybe it should be possible to store a tree without any data ... WHERE ARE features and labels ACTUALLY used?? maybe only for equality test
# 2 trees with different features/labels are not __eq__ even if architecture is same ... this is true if likelihoods are evaluated on the data stored with the tree
# ... but it is an expensive comparison

# MJAS EITHER:
# [1] all-discrete sampling approach uses actual feature values as thresholds 
# which means the proposal mechanism (but not necessarily the tree) would need access to features not labels)
# and the likelihood evaluation also needs access to the len of the features
# ... OR...
# [2] using uniform sampling in each feature we need only store max and min values
# and construct the RJMCMC  diffeomorphism using uniform sampling ... so a Jacobian is needed if we use U[0,1] random number to sample each theshold
# (or no Jacobian if we use U[min value, max value] of the feature)

class Tree:
    def __init__(self, tree = [], leafs = [0], lastAction="none", lastActionHistory = ""): # defaults give valid empty tree that can yield a likelihood!
        # features and labels may be needed for evaluations and proposals but not for the tree class functionality
        #self.features = features
        #self.labels = labels
        self.tree = tree
        self.leafs = leafs
        self.lastAction = lastAction
        self.lastActionHistory = lastActionHistory
        self.evaluations = {} # an evaluation cache, indexed by a unique id e.g. HINTS term index 

    def __eq__(self, x) -> bool: # MJAS WARNING THIS IS NOT TESTING WHETHER 2 TREES ARE EQUIVALENT BECAUSE IT ALSO COMPARES THE ARBITRARY DATA
        #return (x.features == self.features).all() and\
        return(x.tree == self.tree and x.leafs == self.leafs)

    def __str__(self):
        return str(self.tree)

    def __copy__(self):
        # Custom __copy__ to ensure tree and leaf structure are deep copied
        new_tree = Tree(
            #self.features,
            #self.labels,
            copy.deepcopy(self.tree),
            copy.deepcopy(self.leafs),
            "none"
        )
        return new_tree

    @classmethod
    def getProposalType(self):
        return TreeProposal

    @classmethod
    def getTargetType(self):
        return TreeTarget

    @classmethod
    def encode(cls, x):
        raise Exception("NOT IN USE")
        tree = np.array(x.tree).flatten()
        leafs = np.array(x.leafs).flatten()
        last_action = encode_move(x.lastAction)
        tree_dim = len(tree)
        leaf_dim = len(leafs)

        x_new = np.hstack(
            (np.array([tree_dim, leaf_dim, last_action]), tree, leafs)
        )
        return x_new

    @classmethod
    def decode(cls, x, particle):
        raise Exception("NOT IN USE")
        tree_dim = x[0].astype(int)
        leaf_dim = x[1].astype(int)
        last_action = decode_move(x[2].astype(int))
        return Tree(
            #particle.X_train,
            #particle.y_train,
            extract_tree(x[3:(3+tree_dim)]),
            extract_leafs(x[(3+tree_dim):(3+tree_dim+leaf_dim)]),
            last_action
        )

    def depth_of_leaf(self, leaf):
        depth = 0
        for node in self.tree:
            if node[1] == leaf or node[2] == leaf:
                depth = node[5]+1
        return depth
    #
    # only allow prunes that can be reversed
    def prunable_node_indices(self):
        candidates = []
        for i in range(1, len(self.tree)): # cannot prune the root
            node_to_prune = self.tree[i]
            if ((node_to_prune[1] in self.leafs) and (node_to_prune[2] in self.leafs)):
                candidates.append(i)
        return(candidates)
     #   
    # NOT IN USE?
    def grow_leaf(self, features, index, rng=RNG()):
        action = "grow_leaf"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = index
        leaf_to_grow = self.leafs[random_index]

        # generating a random feature
        feature = rng.randomInt(0, len(features[0])-1)
        # generating a random threshold
        threshold = rng.randomInt(0, len(features)-1)
        threshold = (features[threshold, feature])
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth]

        # add the new leafs on the leafs array
        self.leafs.append(max(self.leafs)+1)
        self.leafs.append(max(self.leafs)+1)
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        self.tree.append(node)
        self.evaluations = {} # previous evaluations invalidated by this change
        return self

    def grow(self, features, rng=RNG()):
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        action = "grow"
        self.lastAction = action
        self.lastActionHistory += "G"
        log_fwd = 0.0 # log of the probability of choosing this move given we've reached this point
        #
        nl = len(self.leafs)
        random_index = rng.randomInt(0, nl-1)
        leaf_to_grow = self.leafs[random_index]
        log_fwd -= log(nl)
        #
        # generating a random feature
        nf = features.shape[1]
        feature = rng.randomInt(0, nf-1)
        log_fwd -= log(nf)
        #
        # generating a random threshold
        nfv = features.shape[0]
        ti = rng.randomInt(0, nfv-1)        
        threshold = (features[ti, feature])
        log_fwd -= log(nfv) # log q(x'|x)
        #
        # construct new node
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, leaf_to_grow * 2, leaf_to_grow * 2 + 1, feature,
                threshold, depth]
        #
        # add the new leafs on the leafs array
        self.leafs.append(leaf_to_grow * 2) # correct leaf ID to maximise equivalence
        self.leafs.append(leaf_to_grow * 2 + 1)
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        self.tree.append(node)
        #
        # now what is the log prob associated with the exact reverse move? 
        candidates = self.prunable_node_indices()
        log_rev = -log(len(candidates)) # log q(x|x')
        self.evaluations = {} # previous evaluations invalidated by this change
        return self, log_fwd - log_rev

    def prune(self, features, rng=RNG()):
        action = "prune"
        self.lastAction = action
        self.lastActionHistory += "P"
        
        '''
        MJAS simplfied
        only allow prune actions that delete a single node ... so it is the opposite of Grow
        this makes the Hastings correction so much easier
        '''
        
        # MJAS this code was messy anyway... just use randomInt(1,)
        #random_index = rng.randomInt(0, len(self.tree)-1)
        #node_to_prune = self.tree[random_index]
        #while random_index == 0:
        #    random_index = rng.randomInt(0, len(self.tree)-1)
        #    node_to_prune = self.tree[random_index]
        log_fwd = 0.0 # log of the probability of choosing this move given we've reached this point  
        candidates = self.prunable_node_indices()
        nc = len(candidates)
        random_index = rng.randomInt(0, nc-1)
        index_to_prune = candidates[random_index]
        node_to_prune = self.tree[index_to_prune]
        log_fwd -= log(nc)
        
        # remove the pruned leafs from leafs list and add the node as a
        # leaf
        self.leafs.append(node_to_prune[0])
        self.leafs.remove(node_to_prune[1])
        self.leafs.remove(node_to_prune[2])
        # delete the specific node from the node lists
        del self.tree[index_to_prune]
        # get probability of choosing the reverse (grow operation) - see Grow for details
        log_rev = 0.0
        nl = len(self.leafs)
        log_rev -= log(nl)
        nf = features.shape[1]
        log_rev -= log(nf)
        nfv = len(features)
        log_rev -= log(nfv)
        #
        self.evaluations = {} # previous evaluations invalidated by this change
        return self, log_fwd - log_rev

        '''
            delete_node_indices = []
            i = 0
            for node in self.tree:
                if node_to_prune[1] == node[0] or node_to_prune[2] == node[0]:
                    delete_node_indices.append(node)
                i += 1
            self.tree.remove(node_to_prune)
            for node in delete_node_indices:
                self.tree.remove(node)

            for i in range(len(self.tree)):
                for p in range(1, len(self.tree)):
                    count = 0
                    for k in range(len(self.tree)-1):
                        if self.tree[p][0] == self.tree[k][1] or\
                                self.tree[p][0] == self.tree[k][2]:
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
    '''
    
    # MJAS consider a move that just makes a small change to a threshold
    # this would have much higher acceptance probability so that HINTS always makes progress
    # this would require knowing/finding the position of the existing index in the list of features
    # and changing it by (say) +- a few percent

    
    def change(self, features, rng=RNG()):
        action = "change"
        self.lastAction = action
        self.lastActionHistory += "C"
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have
        chosen and then pick unoformly a node and change their features and
        thresholds
        '''
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        # print(self.tree.__dict__)
        new_feature = rng.randomInt(0, len(features[0])-1)
        new_threshold_index = rng.randomInt(0, len(features)-1) # fixed naming
        node_to_change[3] = new_feature
        node_to_change[4] = features[new_threshold_index, new_feature]
        self.evaluations = {} # previous evaluations invalidated by this change
        return self

    # for the mixing stage its useful to have a small move with a high acceptance prob
    # same as change but (a) keep the same feature (b) limit the change
    # symmetrical
    def adjust(self, features, rng=RNG(), proportion_of_range = 0.05, verbose = False):
        action = "adjust"
        self.lastAction = action
        self.lastActionHistory += "A"
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        # find position of existing threshold in feature list        
        pos = np.searchsorted(features[:, node_to_change[3]], node_to_change[4])
        max_offset = ceil(proportion_of_range * features.shape[0])
        offset = 0
        while offset == 0:
            offset = rng.randomInt(-max_offset, max_offset) # inclusive
        nfv = features.shape[0]
        if pos + offset < 0: # reflecting barrier but with same number of choices
            offset += (max_offset * 2 + 1)
        if pos + offset >= nfv: # reflecting barrier but with same number of choices
            offset -= (max_offset * 2 + 1)            
        new_pos = pos + offset
        new_thresh = features[new_pos, node_to_change[3]]
        if verbose:
            print("existing threshold found at position ", pos, node_to_change[4], features[pos, node_to_change[3]])
            print("new threshold at position ", new_pos, new_thresh)
        node_to_change[4] = new_thresh
        self.evaluations = {} # previous evaluations invalidated by this change      
        return self
    
    # MJAS do we really believe swaps can be useful
    def swap(self, rng=RNG()):
        action = "swap"
        self.lastAction = action
        self.lastActionHistory += "S"
        '''
        need to swap the features and the threshold among the 2 nodes
        '''
        random_index_1 = rng.randomInt(0, len(self.tree)-1)
        random_index_2 = rng.randomInt(0, len(self.tree)-1)
        node_to_swap1 = self.tree[random_index_1]
        node_to_swap2 = self.tree[random_index_2]

        # in case we choose the same node
        while node_to_swap1 == node_to_swap2:
            random_index_2 = rng.randomInt(0, len(self.tree)-1)
            node_to_swap2 = self.tree[random_index_2]

        temporary_feature = node_to_swap1[3]
        temporary_threshold = node_to_swap1[4]

        node_to_swap1[3] = node_to_swap2[3]
        node_to_swap1[4] = node_to_swap2[4]

        node_to_swap2[3] = temporary_feature
        node_to_swap2[4] = temporary_threshold
        
        self.evaluations = {} # previous evaluations invalidated by this change
        return self
