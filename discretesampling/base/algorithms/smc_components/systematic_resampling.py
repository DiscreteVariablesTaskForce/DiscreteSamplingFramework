import numpy as np

def systematic_resampling(x, w, mvrs_rng):
    
    N = len(w)
    #x_new = np.zeros_like(x)
    # x_new = [None] * N
    x_new = []
    w_new = np.ones_like(w) / N
    log_w_new = np.log(w_new)
    p_i = np.zeros_like(w)

    cw = np.cumsum(w)
    
    u = np.random.rand()
    #u = 0.0
    
    for i in range(N):
        j = 0
        
        while cw[j] < (i + u) / N:
            j += 1

        # x_new[i] = x[j]
        x_new.append(x[j])
        #p_i[i] = j

    return x_new, log_w_new
