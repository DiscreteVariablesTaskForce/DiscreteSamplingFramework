import numpy as np
from math import log, inf
import copy
from discretesampling.base.random import RNG
# from discretesampling.base.types import DiscreteVariableProposal

# MJAS note: if prune and grow moves are the same then asymmetry (Hastings correction) is just the ratio of the number of available nodes for pruning/growing [after cancelling a compatible prior]

# class TreeProposal(DiscreteVariableProposal):
# y_train is not used
class TreeProposal:
    def __init__(self, X_train, y_train = None, moves_prob = [0.2, 0.0, 0.2, 0.2, 0.4]):
        # [0.4, 0.1, 0.1, 0.4] # Good for chipman
        # MJAS this code can be tidied further
        #self.moves = ["prune", "swap", "change", "grow", "adjust"]
        #self.reverse_move = {"prune": "grow", "swap":"swap", "change": "change", "grow": "prune", "adjust":"adjust"}
        self.moves_prob = np.array(moves_prob)  # good for Poisson and heart l = 12, and diabetes l = 10
        self.sorted_features = np.sort(X_train, axis = 0)  # full dataset; sort each feature
        #self.y_train = y_train
    #
    #@classmethod
    #def norm(self, tree):
    #    return len(tree.tree)
    #
    #@classmethod
    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    #def heuristic(self, x, y):
    #    return y < x or abs(x-y) < 2
    #
    def corrected_moves_prob(self, num_splits, num_leaves, min_num_splits, max_num_leaves): # if min_splits is zero allow single leaf, empty tree
        moves_prob = copy.deepcopy(self.moves_prob)
        #print(moves_prob)
        if num_splits <= min_num_splits: #can't prune
            moves_prob[0] = 0.0
        if num_splits == 0: # can't change or adjust
            moves_prob[2] = 0.0
            moves_prob[4] = 0.0
        if num_splits < 2 :  # can't swap
            moves_prob[1] = 0.0
        if num_leaves >= max_num_leaves: #can't grow
            moves_prob[3] = 0.0
        moves_prob /= moves_prob.sum()
        return(moves_prob)
    #    
    # original sample function
    def sample(self, start_tree, rng=RNG(), max_num_leaves = 32, verbose = False):
        # initialise the probabilities of each move
        num_leaves  = len(start_tree.leafs) # consistent with the prior on leaves
        num_splits  = len(start_tree.tree)
        min_num_splits = 1
        moves_prob = self.corrected_moves_prob(num_splits, num_leaves, min_num_splits, max_num_leaves)
        
        newTree = copy.deepcopy(start_tree) # ensures a new object copy for cache / diagnostics
        if len(newTree.lastActionHistory) > 0:
            if newTree.lastActionHistory[-1] == 'X': # marked for deletion
                newTree.lastActionHistory = "" # clear it to stop it building up indefinitely
        #
        # MJAS simplified to use the relevant lib func instead of cumul probs
        chosen = rng.randomChoices(range(len(moves_prob)), weights = moves_prob)
        forward_moves_prob = moves_prob[chosen]
        if chosen == 0:
            # prune
            newTree,  hastings = newTree.prune(self.sorted_features, rng=rng)
            contra = 3 # index of opposite action
        #
        elif chosen == 1:
            # swap
            newTree = newTree.swap(rng=rng)
            hastings = 0.0 # symmetrical
            contra = 1
        #
        elif chosen == 2:
            # change
            newTree = newTree.change(self.sorted_features, rng=rng)
            hastings = 0.0 # symmetrical
            contra = 2
        #
        elif chosen == 3:
            # grow
            newTree, hastings = newTree.grow(self.sorted_features, rng=rng)
            contra = 0
        else: 
            # adjust
            newTree = newTree.adjust(self.sorted_features, rng=rng)
            hastings = 0.0
            contra = 4
        #
        reverse_moves_prob = self.corrected_moves_prob(len(newTree.tree), len(newTree.leafs), min_num_splits, max_num_leaves) 
        #
        if verbose:
            print("\n \n Sampled action", newTree.lastAction)
            print(len(start_tree.tree), len(newTree.tree))
            print(round(hastings,3), round(forward_moves_prob,3), round(reverse_moves_prob[contra], 3))
            print("forward moves probs", moves_prob)
            print("reverse moves probs", reverse_moves_prob)
            #print("trees should be different under this equality test [must be False]", newTree == start_tree)
        return newTree,  hastings + log(forward_moves_prob) - log(reverse_moves_prob[contra])

    
    # for HINTS: return log Q(x'\x)/Q(x|x') ... this is -log (proposal ratio) in Green's terminology
    # note this term can be large but will cancel with the prior ratio
    #def proposal_for_HINTS(self, start_tree, rng=RNG(), num_nodes=20, verbose = False):
    #    return(self.sample(start_tree, rng, num_nodes, verbose))
        # this is the slow way of calculating the Hasting's correction, as the priors should cancel and leave only the node count ratio as per Denison
        #forward_moves_prob = self.corrected_moves_prob(num_splits =  len(start_tree.tree), max_splits = num_nodes)
        #reverse_moves_prob = self.corrected_moves_prob(num_splits =  len(new_tree.tree), max_splits = num_nodes)
        #start_revisit = copy.deepcopy(start_tree) ## or use the __copy__ built in - this is just a copy of the original state with reverse action added
        #start_revisit.lastAction = self.reverse_move[new_tree.lastAction]
        #hastings = self.eval(start_tree, new_tree, self.sorted_features, forward_moves_prob) - self.eval(new_tree, start_revisit, self.sorted_features, reverse_moves_prob)
    #    return new_tree, hastings

