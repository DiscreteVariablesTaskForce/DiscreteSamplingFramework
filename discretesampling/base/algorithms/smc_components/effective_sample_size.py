from typing import Union
from discretesampling.base.executor.executor import Executor
import numpy as np


def ess(logw: Union[list[float], np.ndarray], exec: Executor = Executor()) -> float:
    """Compute the Effective Sample Size of the given normalised weights

    Parameters
    ----------
    logw : Union[np.ndarray, list]
        Logged importance normalised weights
    exec : Executor, optional
        Execution engine, by default Executor()

    Returns
    -------
    float
        Effective sample size
    """
    mask = np.invert(np.isneginf(logw))  # mask to filter out any weight = 0 (or -inf in log-scale)

    inverse_neff = np.exp(exec.logsumexp(2*logw[mask]))

    return 1 / inverse_neff
