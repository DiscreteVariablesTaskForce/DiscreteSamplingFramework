import numpy as np


def leaf_class_probs(state):
    """
    Class probabilities for each leaf, as a 2D array of shape (n_leaves, K).
    The rows are in the order of the leaf IDs.
    The columns are in the order of the class labels, 0,...,K-1.
    The probabilities are computed from the counts in the state,
    with a Dirichlet prior given by the problem's alpha parameter.
    """
    problem = state.problem
    ids = np.fromiter(state.counts.keys(), dtype=np.int64, count=len(state.counts))
    block = np.stack([state.counts[leaf] for leaf in ids]).astype(np.float64)
    block += problem.alpha
    block /= block.sum(axis=1, keepdims=True)
    order = np.argsort(ids)
    return ids[order], block[order]


def predict_proba(state, X):
    """
    Class probabilities for each row in X,
    as a 2D array of shape (n_rows, K).
    """
    ids, block = leaf_class_probs(state)
    leaf_of_row = state.route(X)
    pos = np.searchsorted(ids, leaf_of_row)
    return block[pos]


def ensemble_predict_proba(states, X, weights=None):
    """
    Class probabilities for each row in X,
    over all tree particles at the current time,
    averaged over the particle weights if provided.
    """
    states = list(states)
    if not states:
        raise ValueError("no states to predict from")
    if weights is None:
        w = np.full(len(states), 1.0 / len(states))
    else:
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total <= 0:
            raise ValueError("weights sum to zero")
        w = w / total

    out = None
    for weight, state in zip(w, states):
        if weight == 0.0:
            continue
        contribution = predict_proba(state, X) * weight
        out = contribution if out is None else out + contribution
    return out


def predict(states, X, weights=None):
    """
    Returns the maximum a posteriori class label
    from ensemble_predict_proba(states, X, weights).
    """
    return np.argmax(ensemble_predict_proba(states, X, weights), axis=1)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def tree_sizes(states):
    """Node count per tree -- the marginal the distributional tests compare."""
    return np.array([len(s.tree) for s in states], dtype=np.int64)
