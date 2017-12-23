import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Node:
    _left = None
    _right = None
    _middle = None
    _type = None

    def __init__(self, left = None, right = None, middle = None, type = None):
        self._left = left
        self._right = right
        self._middle = middle
        self._type = type


def arg_max_Gain(S, A):
    train_err = float('Inf')
    j = 0
    for i in range(0,A.size):
        Psdemocrat = np.sum(S[:, S.shape[1] - 1] == 'democrat.') / S[:, S.shape[1] - 1].size
        sum = 0
        for k in ('y', 'n', 'u'):
            Sv = S[S[:,A[i]] == k]
            sum = sum + (Sv[:,0].size/S[:,0].size) * C(np.sum(Sv[:, Sv.shape[1] - 1] == 'democrat.') / Sv[:,0].size)
        if Psdemocrat - sum < train_err:
            train_err = Psdemocrat - sum
            j = A[i]
    return j

def C(x):
    return np.min(np.array([x, 1-x]))

def ID3(S , A):
    Y = S[:, S.shape[1] - 1]
    if np.all(Y == 'democrat.'):
        return Node(type='democrat.')
    if np.all(Y == 'republican.'):
        return Node(type='republican.')
    if A.size == 0:
        if np.sum(Y == 'democrat.') > np.sum(Y == 'republican.'):
            return Node(type='democrat.')
        else:
            return Node(type='republican.')
    else:
        j = arg_max_Gain(S, A)
        if np.unique(S[:, j]).size == 1:
            if np.sum(Y == 'democrat.') > np.sum(Y == 'republican.'):
                return Node(type='democrat.')
            else:
                return Node(type='republican.')
        else:
            yes = ID3(S[S[:,j] == 'y'],np.setxor1d(A,np.array([j])))
            no = ID3(S[S[:,j] == 'n'],np.setxor1d(A,np.array([j])))
            un = ID3(S[S[:,j] == 'u'],np.setxor1d(A,np.array([j])))
            return Node(yes, no, un)

        
            





train_set= pd.read_csv('/home/yohai/PycharmProjects/untitled/train_data.csv', sep=' ', index_col=0)


a = train_set.as_matrix()
# print(a.size)
# print(a[a.shape[1], :])
# print(a[:, a.shape[1]-1])
# print(np.arange(a.shape[1]))

n = ID3(a, np.arange(a.shape[1]))
a = 7
