from typing import Union
import numpy as np
from discretesampling.base.executor.executor import Executor


def normalise(logw: Union[list[float], np.ndarray], exec: Executor = Executor()) -> np.ndarray:
    """Normalise importance weights

    Parameters
    ----------
    logw : Union[list[float], np.ndarray]
    array of logged importance weights

    Returns
    -------
    np.ndarray
        log-normalised importance weights

    Notes
    -----
    We have to be careful with -inf values in the log weights
    sometimes. This can happen if we are sampling from a pdf with
    zero probability regions, for example.
    """

    mask = np.invert(np.isneginf(logw))  # mask to filter out any weight = 0 (or -inf in log-scale)

    log_wsum = exec.logsumexp(logw[mask])

    return logw - log_wsum
