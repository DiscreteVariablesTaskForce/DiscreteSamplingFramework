import numpy as np
import heapq

def heapsort(heap):
    return [heapq.heappop(heap) for _ in range(len(heap))]


def kl(log_joint, return_multiplicities=False):
    N = len(log_joint)
    # inputted, unsorted log-likelihood
    logL_s = log_joint

    # init. multiplicity, particle indices, and f_1, f_0
    B_s = np.zeros(N, dtype=np.int32)
    ss = np.arange(0, N)
    f_add = (B_s + 1) * (logL_s - np.log(B_s + 1))
    f_same = np.zeros_like(logL_s)

    # populate heap (negate f_add in order to create max heap)
    heap = []
    for c, idx in zip(f_add, ss):
        heapq.heappush(heap, (-c, idx))
    heap = heapsort(heap)

    # distribute multiplicity
    while np.sum(B_s) < N:
        C_add_max, t_add = heapq.heappop(heap)
        f_same[t_add] = f_add[t_add]
        B_s[t_add] += 1
        f_add[t_add] = (B_s[t_add] + 1) * (logL_s[t_add] - np.log(B_s[t_add] + 1))
        C_add_new = f_add[t_add] - f_same[t_add]
        heapq.heappush(heap, (-C_add_new, t_add))
    newAncestors = np.repeat(np.arange(len(B_s)), B_s)
    
    new_logweights = np.repeat(-np.log(N), N).astype('float32')

    if return_multiplicities:
        return newAncestors, B_s, new_logweights
    else:
        return newAncestors, new_logweights