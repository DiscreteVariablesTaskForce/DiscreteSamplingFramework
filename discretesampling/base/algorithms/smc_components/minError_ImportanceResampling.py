import numpy as np
from discretesampling.base.algorithms.smc_components.importance_resampling_version3 import systematic_resampling
def min_error_importance_resampling(x, w, mvrs_rng , N=None):

    if N is None:
        N = len(w)

    x_new = []
    w_new = np.zeros_like(w)

    if np.sum(w) != 1:
        w = w / np.sum(w)

    S = [i for i in range(N) if w[i] < 1 / N]

    NS = [i for i in range(N) if w[i] > 2 / N]

    if not NS:
        x_new, w_new,_ = systematic_resampling(x, w, mvrs_rng, N)
        return x_new, w_new

    w_tot = np.sum(w[S])
    wwt = w_tot * N
    noninteger = np.ceil(wwt) - wwt

    WS = np.concatenate([w[S], [noninteger / N]])
    Nsmall = int(np.sum(WS) * N)

    WS = WS / np.sum(WS)

    xS = []
    for i in S:
        xS.append(x[i])

    xS.append(x[NS[0]])

    x_small, w_small, _ = systematic_resampling(xS, WS,  mvrs_rng, Nsmall)
    w_small = np.exp(w_small)
    w_small = np.ones(N) / N

    for i in range(Nsmall):
        x_new.append(x_small[i])
        w_new[i] = w_small[i]

    NSindices = np.setdiff1d(np.arange(N), S)

    wNS = w[NS[0]]
    wNS = wNS * N - noninteger
    ww = np.zeros_like(w)
    for i in NSindices:
        ww[i] = w[i]

    if NS:
        i = next(iter(NS))  # Get the first element from the set
        ww[i] = wNS / N

    ww[NSindices] = ww[NSindices] / np.sum(ww[NSindices])

    xnotS=[]
    for i in NSindices:
        xnotS.append(x[i])

    x_NS, w_NS = importance_resampling(xnotS, ww[NSindices], mvrs_rng, N - Nsmall)
    w_NS = np.exp(w_NS)

    w_NS = w_NS * (N - Nsmall) / N

    x_new[Nsmall:] = x_NS
    w_new[Nsmall:] = w_NS

    log_w_new = np.log(w_new)

    return x_new, log_w_new


########################################################################
def importance_resampling(x, w, mvrs_rng, N=None):

    if N is None:
        N = len(w)

    _, _, nc = min_error_continuous_state_resampling(x, w, mvrs_rng, N)

    quantisedweights = nc/N

    x_new, _, ncopies = min_error_continuous_state_resampling(x, quantisedweights, mvrs_rng, N)

    ww = w / quantisedweights
    w_new = np.repeat(ww, ncopies, axis=0)

    w_new /= np.sum(w_new)

    log_w_new = np.log(w_new)

    return x_new, log_w_new

########################################################################
def min_error_continuous_state_resampling(x, w,  mvrs_rng , N=None):

    if N is None:
        N = len(w)

    N = int(N)

    x_new = []
    w_new = np.ones(N) / N
    log_w_new = np.log(w_new)

    i = 0
    nc = np.floor(N * w).astype(int) #number of copies

    for j in range(len(nc)):
        for k in range(nc[j]):
            i += 1
            x_new.append(x[j])

    NW = N * w
    non_integer_parts = NW - np.floor(NW)
    sorted_indices = np.argsort(non_integer_parts)[::-1]
    Nr = N - np.sum(nc) #Number of the rest of the samples
    sorted_indices = sorted_indices[:Nr]
    for i in sorted_indices:
        x_new.append(x[i])

    nc[sorted_indices] += 1

    return x_new, log_w_new, nc

