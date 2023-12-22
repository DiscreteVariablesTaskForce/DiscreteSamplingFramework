import copy
import numpy as np
from itertools import chain

#insert new list of values at a given index

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


def find_rand(l, q):
    if len(l) == 1:
        return 0
    else:
        inds = np.where(l>q)
        return inds[0][0]

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

