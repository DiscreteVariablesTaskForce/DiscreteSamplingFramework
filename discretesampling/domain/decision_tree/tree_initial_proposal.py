import math
from discretesampling.base.types import DiscreteVariableInitialProposal
from discretesampling.base.random import RNG
from discretesampling.domain.decision_tree import Tree
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


# MJAS Can we eliminate this... it's only used for the start state

# class TreeInitialProposal(DiscreteVariableInitialProposal):
class TreeInitialProposal:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train # MJAS not needed but leaving this in for the Random Forest stuff
        # self.rng = rng

    # def sample(self, rng):
    #     leafs = [1, 2]

    #     feature = rng.randomInt(0, len(self.X_train[0])-1)
    #     threshold = rng.randomInt(0, len(self.X_train)-1)
    #     tree = [[0, 1, 2, feature, self.X_train[threshold, feature],0]]
    #     return Tree(self.X_train, self.y_train, tree, leafs)

    # MJAS why bother with this ... just create a single leaf instead?
    # ... and would be better to use the tree Grow() etc rather than duplicate code!  
    # leaving as is for now
    def sample(self, rng=RNG(), target=None):
        leafs = [2, 3] # MJAS corrected so child nodes are 2 * parent and 2 * parent +1 (start counting at 1)
        feature = rng.randomInt(0, len(self.X_train[0])-1)
        threshold = rng.randomInt(0, len(self.X_train)-1)
        tree = [[1, 2, 3, feature, self.X_train[threshold, feature], 0]]
        init_tree = Tree(tree, leafs)

        if target is None:
            return init_tree

        i = 0
        while i < len(leafs):
            u = rng.uniform()
            prior = math.exp(target.evaluatePrior(init_tree))
            # print("tree before: ", init_tree)
            if u < prior:
                init_tree = init_tree.grow_leaf(leafs.index(leafs[i]), rng)
                leafs = init_tree.leafs
            else:
                i += 1
            
        return init_tree

    def eval(self, x, target=None):
        num_features = len(self.X_train[0])
        num_thresholds = len(self.X_train)
        if target is None:
            return -math.log(num_features) - math.log(num_thresholds)
        else:
            return -math.log(num_features) - math.log(num_thresholds) + target.evaluatePrior(x)
        
    def sample_RF_proposals(self, rng=RNG(), target=None):
        init_trees=[]
        # Train a Random Forest Regression model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)

        # Function to extract tree structure
        def extract_tree_structure(tree, feature_names):
            tree_structure = []
            leaf_nodes = []

            def recurse(node, depth):
                if tree.children_left[node] != tree.children_right[node]:
                    tree_structure.append([
                        node,
                        tree.children_left[node],
                        tree.children_right[node],
                        feature_names[tree.feature[node]],
                        tree.threshold[node],
                        depth
                    ])
                    recurse(tree.children_left[node], depth + 1)
                    recurse(tree.children_right[node], depth + 1)
                else:
                    leaf_nodes.append(node)

            recurse(0, 0)
            return tree_structure, leaf_nodes

        # Extract tree structures for all trees in the random forest
        forest_structure = []
        forest_leaf_nodes = []
        print(rf.estimators_)
        for estimator in rf.estimators_:
            
            tree_structure, leaf_nodes = extract_tree_structure(estimator.tree_, feature_names=list(range(self.X_train.shape[1])))
            forest_structure.append(tree_structure)
            forest_leaf_nodes.append(leaf_nodes)

        # # Display the tree structures
        # for i, tree_structure in enumerate(forest_structure):
        #     print(f"Tree {i+1}:")
        #     for node in tree_structure:
        #         print(node)
        #     print()

        # # Display the leaf nodes for each tree
        # for i, tree_leaf_nodes in enumerate(forest_leaf_nodes):
        #     print(f"Leaf nodes for Tree {i+1}:")
        #     print(tree_leaf_nodes)
        #     print()
            
        for i in range(len(forest_structure)):
            init_trees.append(Tree(self.X_train, self.y_train, forest_structure[i], forest_leaf_nodes[i]))
        print("passed this")
        return init_trees
    
    
    def sample_RF_proposals2(self, rng=RNG()):
        loc_n = len(rng)
        init_trees=[]
        

        # Train loc_n Random Forest Regression models
        rfs = []
        rfs = [RandomForestRegressor(n_estimators=1, random_state=rng[i].seed) for i in range(loc_n)]
        for i in range(loc_n):
            rfs[i].fit(self.X_train, self.y_train)

        # Function to extract tree structure
        def extract_tree_structure(tree, feature_names):
            tree_structure = []
            leaf_nodes = []

            def recurse(node, depth):
                if tree.children_left[node] != tree.children_right[node]:
                    tree_structure.append([
                        node,
                        tree.children_left[node],
                        tree.children_right[node],
                        feature_names[tree.feature[node]],
                        tree.threshold[node],
                        depth
                    ])
                    recurse(tree.children_left[node], depth + 1)
                    recurse(tree.children_right[node], depth + 1)
                else:
                    leaf_nodes.append(node)

            recurse(0, 0)
            return tree_structure, leaf_nodes

        # Extract tree structures for all trees in the random forest
        forest_structure = []
        forest_leaf_nodes = []
        #for estimator in rf.estimators_:
        for i in range(loc_n):
            estimator = rfs[i].estimators_[0]
            tree_structure, leaf_nodes = extract_tree_structure(estimator.tree_, feature_names=list(range(self.X_train.shape[1])))
            forest_structure.append(tree_structure)
            forest_leaf_nodes.append(leaf_nodes)

        # # Display the tree structures
        # for i, tree_structure in enumerate(forest_structure):
        #     print(f"Tree {i+1}:")
        #     for node in tree_structure:
        #         print(node)
        #     print()

        # # Display the leaf nodes for each tree
        # for i, tree_leaf_nodes in enumerate(forest_leaf_nodes):
        #     print(f"Leaf nodes for Tree {i+1}:")
        #     print(tree_leaf_nodes)
        #     print()
            
        for i in range(len(forest_structure)):
            init_trees.append(Tree(self.X_train, self.y_train, forest_structure[i], forest_leaf_nodes[i]))
        print("passed this")
        return init_trees
