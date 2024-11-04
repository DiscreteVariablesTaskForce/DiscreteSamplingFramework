import numpy as np
from scipy.special import logsumexp

from discretesampling.base.executor.executor import Executor
from discretesampling.base.executor.executor_MPI import Executor_MPI


def normalise(logw, exec=Executor()):
    """
    Description
    -----------
    Normalise importance weights. Note that we remove the mean here
        just to avoid numerical errors in evaluating the exponential.
        We have to be careful with -inf values in the log weights
        sometimes. This can happen if we are sampling from a pdf with
        zero probability regions, for example.

    Parameters
    ----------
    logw : array of logged importance weights

    Returns
    -------
    - : array of log-normalised importance weights

    """

    m = np.invert(np.isneginf(logw))  # mask to filter out any weight = 0 (or -inf in log-scale)

    log_wsum = exec.logsumexp(np.where(m, logw, 0))

    return logw - log_wsum
