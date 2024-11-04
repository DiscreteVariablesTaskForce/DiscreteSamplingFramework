import numpy as np


def ess(logw, exec):
    """
    Description
    -----------
    Computes the Effective Sample Size of the given normalised weights

    Parameters
    ----------
    logw : array of logged importance normalised weights

    Returns
    -------
    double scalar : Effective Sample Size

    """

    mask = np.invert(np.isneginf(logw))  # mask to filter out any weight = 0 (or -inf in log-scale)

    logw = np.array([logw[i] for i in range(len(logw)) if mask[i]])
    inverse_neff = np.exp(exec.logsumexp(2*logw))

    return 1 / inverse_neff
