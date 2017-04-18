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


print(calcLS(sample(4,0.1)).shape)