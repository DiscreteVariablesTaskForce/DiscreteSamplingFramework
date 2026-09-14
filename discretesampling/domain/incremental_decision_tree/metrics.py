import numpy as np


def _alpha_of(state):
    """
    The Dirichlet-Multinomial concentration behind a state.

    A live IncBDTree reaches it through the problem it was built from. A state
    read back off disk (states.TreeStoreView) has no problem to consult -- the
    training data is not part of what was stored -- so it carries alpha itself.
    """
    problem = getattr(state, 'problem', None)
    return problem.alpha if problem is not None else state.alpha


def route_rows(rows, X):
    """
    The leaf id each row of X reaches, for a tree given as its
    [id, left, right, feat, thr, depth] rows alone.

    IncBDTree.route is this same walk over a live tree; this is the version a
    stored state can run, holding only the rows. Routed a node at a time rather
    than a row at a time: every row visits the same nodes either way, but the
    comparison at each node is one numpy operation over the rows that reached
    it instead of a Python loop over rows.
    """
    X = np.asarray(X)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    node_map = {int(r[0]): r for r in rows}
    out = np.empty(len(X), dtype=np.int64)
    pending = [(0, np.arange(len(X)))]
    while pending:
        nid, idx = pending.pop()
        if idx.size == 0:
            continue
        row = node_map.get(nid)
        if row is None:                     # a leaf: these rows stop here
            out[idx] = nid
            continue
        go_left = X[idx, int(row[3])] < row[4]
        pending.append((int(row[1]), idx[go_left]))
        pending.append((int(row[2]), idx[~go_left]))
    return out


def _sorted_leaf_counts(state):
    """(leaf ids, (n_leaves, K) float64 class counts), sorted by leaf id."""
    counts = state.counts
    ids = getattr(counts, 'ids', None)
    if ids is not None:
        # A stored state keeps its leaf ids sorted alongside the counts block,
        # so neither the per-leaf lookup nor the sort below is needed.
        return ids, np.asarray(counts.block, dtype=np.float64)
    ids = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    block = np.stack([counts[leaf] for leaf in ids]).astype(np.float64)
    order = np.argsort(ids)
    return ids[order], block[order]


def leaf_class_probs(state):
    """
    Class probabilities for each leaf, as a 2D array of shape (n_leaves, K).
    The rows are in the order of the leaf IDs.
    The columns are in the order of the class labels, 0,...,K-1.
    The probabilities are computed from the counts in the state,
    with a Dirichlet prior given by the problem's alpha parameter.
    """
    ids, block = _sorted_leaf_counts(state)
    block = block + _alpha_of(state)
    block /= block.sum(axis=1, keepdims=True)
    return ids, block


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


# ------------------- classification metrics ------------------- #
#
# Written against numpy rather than sklearn.metrics because they are called
# once per sampler iteration: on a run of any length the per-call overhead of
# the sklearn entry points, which validate and re-derive the label set every
# time, costs more than the arithmetic. The label set is fixed by the problem,
# so it is passed in as num_classes instead of rediscovered.

def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Rows are true classes, columns predicted, as a (K, K) integer array.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1
    K = int(num_classes)
    return np.bincount(y_true * K + y_pred,
                       minlength=K * K).reshape(K, K)


def _safe_divide(num, den):
    """num/den, defining 0/0 as 0 -- a class with no predictions has no
    precision and a class with no support has no recall, and neither is a
    reason to put a nan through the rest of a run's statistics."""
    out = np.zeros(len(num), dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def precision_recall_f1(cm):
    """
    Per-class precision, recall and F1, read off a confusion matrix.
    """
    tp = np.diag(cm).astype(np.float64)
    precision = _safe_divide(tp, cm.sum(axis=0).astype(np.float64))
    recall = _safe_divide(tp, cm.sum(axis=1).astype(np.float64))
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def balanced_accuracy(cm):
    """
    Mean recall over the classes that appear in y_true. Classes with no support
    are left out rather than counted as zero, which would otherwise make the
    figure depend on how many classes the problem declares.
    """
    support = cm.sum(axis=1)
    present = support > 0
    if not present.any():
        return float('nan')
    _, recall, _ = precision_recall_f1(cm)
    return float(recall[present].mean())


_PROB_FLOOR = 1e-15


def log_loss(y_true, proba, eps=_PROB_FLOOR):
    """
    Mean negative log probability of the true class. Unlike accuracy this reads
    the whole predictive distribution, so it separates a sampler that is right
    for the right reasons from one that is right and unsure.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    p = proba[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p, eps, 1.0))))


def brier_score(y_true, proba):
    """
    Mean squared error of the predicted distribution against the one-hot truth,
    summed over classes. Bounded, unlike log_loss, so it survives a sampler that
    puts zero mass on an observed class.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    sq = np.sum(proba * proba, axis=1)
    return float(np.mean(sq - 2.0 * proba[np.arange(len(y_true)), y_true] + 1.0))


def classification_metrics(y_true, proba, num_classes=None, prefix=""):
    """
    Every scalar metric for one prediction, from a single predict_proba pass.

    Returns a flat dict, so a per-iteration record is one call and the caller
    can stack the results straight into columns. `prefix` names the split, e.g.
    prefix="test_" gives test_accuracy, test_log_loss, and so on. The confusion
    matrix is returned under prefix + "confusion" alongside the scalars.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.argmax(proba, axis=1)
    if num_classes is None:
        num_classes = proba.shape[1]
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return _metrics_from(cm, log_loss(y_true, proba), brier_score(y_true, proba),
                         prefix)


def _metrics_from(cm, log_loss_value, brier_value, prefix):
    """The flat metrics dict, given the confusion matrix and the two metrics
    that read the probabilities rather than only the predicted labels."""
    precision, recall, f1 = precision_recall_f1(cm)
    support = cm.sum(axis=1) > 0
    return {
        prefix + 'accuracy': float(np.trace(cm) / max(cm.sum(), 1)),
        prefix + 'balanced_accuracy': balanced_accuracy(cm),
        prefix + 'macro_precision': float(precision[support].mean()) if support.any() else float('nan'),
        prefix + 'macro_recall': float(recall[support].mean()) if support.any() else float('nan'),
        prefix + 'macro_f1': float(f1[support].mean()) if support.any() else float('nan'),
        prefix + 'log_loss': float(log_loss_value),
        prefix + 'brier': float(brier_value),
        prefix + 'confusion': cm,
    }


# ------------------- per-tree metrics, and their weighted mean ------------------- #
#
# Every row reaching a leaf gets that leaf's predictive distribution, so a
# tree's metrics on a set of rows depend only on how many rows of each class
# reach each leaf. Scoring those (n_leaves, K) tallies is exact and skips the
# (n_rows, K) probability matrix; on the rows a tree was fitted to, the tallies
# are its stored leaf counts, and nothing needs routing at all.

def _leaf_metrics(probs, tallies, prefix):
    """
    classification_metrics from `probs`, each leaf's (n_leaves, K) predictive
    distribution, and `tallies`, the (n_leaves, K) count of rows of each true
    class that reached it.
    """
    tallies = np.asarray(tallies, dtype=np.int64)
    K = probs.shape[1]
    cm = tallies.T @ np.eye(K, dtype=np.int64)[np.argmax(probs, axis=1)]
    n = max(int(tallies.sum()), 1)
    log_p = np.log(np.clip(probs, _PROB_FLOOR, 1.0))
    sq = np.sum(probs * probs, axis=1, keepdims=True)
    return _metrics_from(cm,
                         -np.sum(tallies * log_p) / n,
                         np.sum(tallies * (sq - 2.0 * probs + 1.0)) / n,
                         prefix)


def leaf_tallies(state, X, y):
    """
    (n_leaves, K) counts of the rows of (X, y) reaching each leaf, by true
    class, in the leaf-id order of leaf_class_probs. On the data the tree was
    fitted to, these are its leaf counts.
    """
    ids, counts = _sorted_leaf_counts(state)
    L, K = counts.shape
    pos = np.searchsorted(ids, state.route(X))
    y = np.asarray(y, dtype=np.int64)
    return np.bincount(pos * K + y, minlength=L * K).reshape(L, K)


def state_metrics(state, X, y, prefix=""):
    """classification_metrics for one tree on (X, y)."""
    _, probs = leaf_class_probs(state)
    return _leaf_metrics(probs, leaf_tallies(state, X, y), prefix)


def fitted_metrics(state, prefix=""):
    """
    classification_metrics for one tree on the data it was fitted to, read off
    its leaf counts without routing anything: state_metrics(state, X_fit,
    y_fit) up to float rounding, at O(n_leaves x K) rather than O(n_rows x depth).
    """
    _, probs = leaf_class_probs(state)
    _, counts = _sorted_leaf_counts(state)
    return _leaf_metrics(probs, counts, prefix)


def weighted_mean(metrics, weights):
    """
    Key-by-key weighted mean of per-tree metric dicts, for linear-scale weights
    summing to one. Scalars come back as floats, and the confusion matrix as
    the expected (K, K) confusion matrix.
    """
    w = np.asarray(weights, dtype=np.float64)
    out = {}
    for key in metrics[0]:
        mean = np.tensordot(w, np.asarray([m[key] for m in metrics],
                                          dtype=np.float64), axes=1)
        out[key] = float(mean) if mean.ndim == 0 else mean
    return out


def evaluate(states, X, y, weights=None, prefix=""):
    """
    The weighted mean of each state's own metrics -- one MCMC state, or a
    weighted SMC particle set. `weights` are on the linear scale and are
    normalised here; leave them out for an equally weighted set.

    This is E_pi[metric], the posterior expectation of each metric, not a
    metric read off the pi-averaged predictive distribution
    (ensemble_predict_proba) -- the two differ in general, since e.g. the
    argmax of an average isn't the average of the argmaxes. A single MCMC
    state at weight 1 is the trivial case: that state's own metrics.
    """
    states = list(states)
    if not states:
        raise ValueError("no states to evaluate")
    if weights is None:
        w = np.full(len(states), 1.0 / len(states))
    else:
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total <= 0:
            raise ValueError("weights sum to zero")
        w = w / total
    keep = np.flatnonzero(w > 0)
    return weighted_mean([state_metrics(states[i], X, y, prefix) for i in keep],
                         w[keep])
