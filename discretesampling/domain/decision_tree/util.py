ID = 0
LCHILD = 1
RCHILD = 2
FEATURE = 3
THRESHOLD = 4


def find_leaf_for_datum(x, datum):
    current_node = x.tree[0]

    # make sure that we are not in leafs. current_node[0] is the node
    while current_node[ID] not in x.leafs:
        if datum[current_node[FEATURE]] > current_node[THRESHOLD]:
            if current_node[RCHILD] in x.leafs:
                leaf = current_node[RCHILD]
                break
            else:
                current_node = x.tree[current_node[RCHILD]]
        else:
            if current_node[LCHILD] in x.leafs:
                leaf = current_node[LCHILD]
                break
            else:
                current_node = x.tree[current_node[LCHILD]]
    return leaf
