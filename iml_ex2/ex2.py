import numpy as np
import matplotlib.pylab as plt


def sample(m, epsilon):
    samp = np.random.uniform(0,1,(m, 2))
    omega_star = np.array([0.6, -1])
    y = np.inner(samp, omega_star) + np.random.normal(0, epsilon, m)
    return np.c_[samp, np.transpose(y)]



def calcLS(S):
    w_low = -3
    w_up = 3
    mesh_range = np.arange(w_low, w_up, (w_up - w_low)/200)
    ws = np.array(np.meshgrid(mesh_range, mesh_range ))
    y = S[:, 2:]
    d = np.empty((200, 200))
    for i in range(200):
        for j in range(200):
            d[i][j] = 0.5*(np.sum(np.abs(np.dot(S[:, :2], np.array([ws[:,i,j]]).T) - y)**2))
    return d

def Ls(S,w):
    return (np.sum(np.abs(np.array([np.dot(S[:, :2], w.T)]).T - S[:, 2:])**2)*0.5)/S.shape[0]

def calcGradient(S,w):
    # epsilon = 0.00001
    # dx = (Ls(S, np.array([w[0] + epsilon, w[1]])) - Ls(S,w))/ epsilon
    # dy = (Ls(S, np.array([w[0] , w[1] + epsilon])) - Ls(S, w)) / epsilon
    dx =  (np.sum(np.array([np.dot(S[:, :2], w.T)]).T - S[:, 2:])*w[0])/S.shape[0]
    dy = (np.sum(np.array([np.dot(S[:, :2], w.T)]).T - S[:, 2:]) * w[1]) / S.shape[0]

    return np.array([dx,dy])










