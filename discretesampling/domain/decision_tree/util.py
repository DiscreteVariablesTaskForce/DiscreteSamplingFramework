def encode_move(last):
    if last == "grow":
        return 0
    elif last == "prune":
        return 1
    elif last == "swap":
        return 2
    elif last == "change":
        return 3
    else:
        return -1
        
def decode_move(code):
    if code == 0:
        return "grow"
    elif code == 1:
        return "prune"
    elif code == 2:
        return "swap"
    elif code == 3:
        return "change"
    else:
        return ""

def extract_tree(tree):
    return [tree[i:i+4].astype(int).tolist() + [tree[i + 4]] + [tree[i + 5].astype(int)] for i in range(0, len(tree.tolist()), 6)]

def extract_leafs(leaves):
    return leaves.tolist()

