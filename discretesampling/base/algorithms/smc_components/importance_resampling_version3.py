import numpy as np
def importance_resampling_v3(x, w, mvrs_rng , N=None):

    if N is None:
        N = len(w)

    #x_new = np.zeros_like(x)
    x_new = []
    w_new = np.zeros_like(w)

    if np.sum(w) != 1:
        w = w / np.sum(w)

    S = [i for i in range(N) if w[i] < 1 / N]

    NS = [i for i in range(N) if w[i] > 2 / N]

    if not NS:
        x_new, w_new = systematic_resampling(x, w, mvrs_rng, N)
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

    x_small, w_small = systematic_resampling(xS, WS, mvrs_rng, Nsmall)
    w_small = np.exp(w_small)
    w_small = np.ones(N) / N

    for i in range(Nsmall):
        #x_new[i] = x_small[i]
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

    x_new_s, w_new_s = systematic_resampling(x, w, mvrs_rng, N)

    w_new_s = np.exp(w_new_s)
    quantisedweights = np.zeros_like(x)

    for i in range(len(x)):
        for j in range(len(x_new_s)):
            if x_new_s[j] == x[i]:
                quantisedweights[i] += w_new_s[j]

    x_new, _ = systematic_resampling(x, quantisedweights, mvrs_rng, N)


    w_new = np.zeros_like(x_new)

    for i in range(len(x_new)):
        for j in range(len(x)):
            if x_new[i] == x[j]:
                w_new[i] = w[j] / quantisedweights[j]

    w_new /= np.sum(w_new)

    #log_w_new = np.log(w_new)
    log_w_new = []
    for k in w_new:
        log_w_new.append(np.log(k))


    return x_new, log_w_new

########################################################################

def systematic_resampling(x, w,  mvrs_rng , N=None):
    if N is None:
        N = len(w)

    N = int(N)

    x_new = []
    w_new = np.ones(N) / N
    log_w_new = np.log(w_new)

    cw = np.cumsum(w)

    u = mvrs_rng.uniform()

    for i in range(N):
        j = 0

        while cw[j] < (i + u) / N:
            j += 1

        x_new.append(x[j])

    return x_new, log_w_new
