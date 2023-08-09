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


def extract_tree(encoded_tree):
    return [
        encoded_tree[i:i+3].astype(int).tolist()  # nodes
        + [encoded_tree[i+3].astype(int)]  # feature
        + [encoded_tree[i+4]]  # threshold
        + [encoded_tree[i+5].astype(int)]  # depth
        for i in range(0, len(encoded_tree.tolist()), 6)
    ]


def extract_leafs(leafs):
    return leafs.astype(int).tolist()
