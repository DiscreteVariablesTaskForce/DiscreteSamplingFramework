from ...base.random import RandomInt
from ...base import types
from .tree_distribution import TreeProposal
from .tree_target import TreeTarget


class Tree(types.DiscreteVariable):
    def __init__(self, X_train, y_train, tree, leafs):
        self.X_train = X_train
        self.X_train.flags.writeable = False
        self.y_train = y_train
        self.y_train.flags.writeable = False
        self.tree = tuple(tuple(x) for x in tree)
        self.leafs = tuple(leafs)


    def __eq__(self, x) -> bool:
        return (x.X_train == self.X_train).all() and\
                (x.y_train == self.y_train).all() and\
                x.tree == self.tree and x.leafs == self.leafs

    def __hash__(self) -> int:
        return hash((
            self.X_train.tostring(),
            self.y_train.tostring(),
            self.tree,
            self.leafs
        ))

    def __str__(self):
        return str(self.tree)

    @classmethod
    def getProposalType(self):
        return TreeProposal

    @classmethod
    def getTargetType(self):
        return TreeTarget

    def grow(self):
        action = "grow"
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = RandomInt(0, len(self.leafs)-1).eval()
        new_tree = list(list(x) for x in self.tree)
        new_leafs = list(self.leafs)

        leaf_to_grow = new_leafs[random_index]

        # generating a random faeture
        feature = RandomInt(0, len(self.X_train[0])-1).eval()
        # generating a random threshold
        threshold = RandomInt(0, len(self.X_train)-1).eval()
        threshold = (self.X_train[threshold, feature])

        node = [leaf_to_grow, max(new_leafs)+1, max(new_leafs)+2, feature,
                threshold]

        # add the new leafs on the leafs array
        new_leafs.append(max(new_leafs)+1)
        new_leafs.append(max(new_leafs)+1)
        # delete from leafs the new node
        new_leafs.remove(leaf_to_grow)
        new_tree.append(node)

        return Tree(self.X_train, self.y_train, new_tree, new_leafs)

    def prune(self):
        action = "prune"
        '''
        For example when we have nodes 0,1,2 and leafs 3,4,5,6 when we prune
        we take the leafs 6 and 5 out, and the
        node 2, now becomes a leaf.
        '''

        new_leafs = list(self.leafs)
        new_tree = list(list(x) for x in self.tree)

        random_index = RandomInt(0, len(self.tree)-1).eval()
        node_to_prune = new_tree[random_index]
        while random_index == 0:
            random_index = RandomInt(0, len(self.tree)-1).eval()
            node_to_prune = new_tree[random_index]

        if (node_to_prune[1] in new_leafs) and\
                (node_to_prune[2] in new_leafs):
            # remove the pruned leafs from leafs list and add the node as a
            # leaf
            new_leafs.append(node_to_prune[0])
            new_leafs.remove(node_to_prune[1])
            new_leafs.remove(node_to_prune[2])
            # delete the specific node from the node lists
            del new_tree[random_index]
        else:
            delete_node_indices = []
            i = 0
            for node in new_tree:
                if node_to_prune[1] == node[0] or node_to_prune[2] == node[0]:
                    delete_node_indices.append(node)
                i += 1
            new_tree.remove(node_to_prune)
            for node in delete_node_indices:
                new_tree.remove(node)

            for i in range(len(new_tree)):
                for p in range(1, len(new_tree)):
                    count = 0
                    for k in range(len(new_tree)-1):
                        if new_tree[p][0] == new_tree[k][1] or\
                                new_tree[p][0] == new_tree[k][2]:
                            count = 1
                    if count == 0:
                        new_tree.remove(new_tree[p])
                        break

        new_leafs = []
        for node in new_tree:
            count1 = 0
            count2 = 0
            for check_node in new_tree:
                if node[1] == check_node[0]:
                    count1 = 1
                if node[2] == check_node[0]:
                    count2 = 1

            if count1 == 0:
                new_leafs.append(node[1])

            if count2 == 0:
                new_leafs.append(node[2])

        return Tree(self.X_train, self.y_train, new_tree, new_leafs)

    def change(self):
        action = "change"
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have
        chosen and then pick unoformly a node and change their features and
        thresholds
        '''
        random_index = RandomInt(0, len(self.tree)-1).eval()
        new_tree = list(list(x) for x in self.tree)
        new_leafs = list(self.leafs)
        node_to_change = new_tree[random_index]
        new_feature = RandomInt(0, len(self.X_train[0])-1).eval()
        new_threshold = RandomInt(0, len(self.X_train)-1).eval()

        node_to_change[3] = new_feature
        node_to_change[4] = self.X_train[new_threshold, new_feature]

        return Tree(self.X_train, self.y_train, new_tree, new_leafs)

    def swap(self):
        action = "swap"
        '''
        need to swap the features and the threshold among the 2 nodes
        '''
        random_index_1 = RandomInt(0, len(self.tree)-1).eval()
        random_index_2 = RandomInt(0, len(self.tree)-1).eval()
        new_tree = list(list(x) for x in self.tree)
        new_leafs = list(self.leafs)
        node_to_swap1 = new_tree[random_index_1]
        node_to_swap2 = new_tree[random_index_2]

        # in case we choose the same node
        while node_to_swap1 == node_to_swap2:
            random_index_2 = RandomInt(0, len(new_tree)-1).eval()
            node_to_swap2 = new_tree[random_index_2]

        temporary_feature = node_to_swap1[3]
        temporary_threshold = node_to_swap1[4]

        node_to_swap1[3] = node_to_swap2[3]
        node_to_swap1[4] = node_to_swap2[4]

        node_to_swap2[3] = temporary_feature
        node_to_swap2[4] = temporary_threshold

        return Tree(self.X_train, self.y_train, new_tree, new_leafs)
