import copy
import numpy as np
import math

def insert_pair(l, ins, ind):

    update_list = []
    for i in range(len(l)+1):
        if i == ind or i == ind+1:
            update_list.append(0)
        elif i < ind:
            update_list.append(l[i])
        else:
            update_list.append(l[i-1])

    update_list[ind]+=ins[0]
    update_list[ind+1]+=ins[1]

    return update_list

def find_rand(l, u):
    return np.argwhere(u<l)[0][0]

def assign_from_pmf(pmf):
    cdf = np.cumsum(pmf)
    if np.abs(cdf[-1]-1) >10**-8:
        print('Error: This is not a PMF, it sums to {}'.format(cdf[-1]))
        return 0
    else:
        u = np.random.uniform(0,1)
        return np.argwhere(u<cdf)[0][0]



#check that a list is sorted
def check_ordered(l):
    l_check = sorted(copy.copy(l))
    return all(l[i] == l_check[i] for i in range(len(l)))

def normalise(l):
    s = sum(l)
    for i in range(len(l)):
        l[i] = l[i]/s
    return l

def allocation_counter(l, inds):
    return [l.count(i) for i in inds]

def kill_list(l, ind, renorm=False):
    if not renorm:
        del l[ind]
        return l
    else:
        del l[ind]
        return [i/sum(l) for i in l]

def bangit(n):
    r = copy.copy(n)
    for i in range(1, n):
        print('{} x {}'.format(n, n-i))
        n*=(r-i)

    return n

def logbang(n):
    logn = math.log(n)
    r = copy.copy(n)

    for i in range(1,n):
        logn+=math.log(r-i)

    return logn

def matchlist(l1, l2):
    if len(l1) <= len(l2):
        short = l1
        long = l2
    else:
        short = l2
        long = l1

    short_set = set(short)

    c = 0

    for i in long:
        if i not in short_set:
            c+=1

    return c

def logsumexp(x):
    c = max(x)
    return c + math.log(np.sum(np.exp(x-c)))