import numpy as np
from discretesampling.base.algorithms.smc_components.systematic_resampling import systematic_resampling

def importance_resampling(x, w, mvrs_rng):
    N = len(w)
    x_new_s, w_new_s = systematic_resampling(x,w, mvrs_rng)
    w_new_s = np.exp(w_new_s)
    quantisedweights = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if x_new_s[j] == x[i]:
                quantisedweights[i] += w_new_s[j]

    x_new, _ = systematic_resampling(x, quantisedweights, mvrs_rng)

    w_new = np.zeros_like(w)

    for i in range(N):
        for j in range(N):
            if x_new[i] == x[j]:
                w_new[i] = w[j] / quantisedweights[j]

    w_new /= np.sum(w_new)

    log_w_new = np.log(w_new)

    return x_new, log_w_new
