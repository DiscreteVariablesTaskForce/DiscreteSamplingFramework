from .tree_distribution import TreeProposal
import copy

class SpiderTreeProposal(TreeProposal):
    def __init__(self, tree):
        self.X_train = tree.X_train
        self.y_train = tree.y_train
        self.tree = copy.deepcopy(tree)
        self.moves_prob = [] # function of pheromones