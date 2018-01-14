import numpy as np
import matplotlib.pyplot as plt

def dtft(x):
    x_ft = np.zeros(x.shape, dtype='complex64')
    for k in range(x.shape[0]):
        for n in range(x.shape[0]):
            x_ft[k] += x[n]*np.exp(-1j*2*np.pi*k*n/x.shape[0])
    return x_ft

def invers_dtft(x):
    x_ft = np.zeros(x.shape, dtype='complex64')
    for k in range(x.shape[0]):
        for n in range(x.shape[0]):
            x_ft[k] += x[n] * np.exp(1j * 2 * np.pi * k * n / x.shape[0])
    return x_ft/x.shape[0]


def cyc_cunv(x , y):
    sol = np.zeros(x.shape)
    for n in range(x.shape[0]):
        for m in range(x.shape[0]):
            sol[n] += x[m]*y[np.mod((n-m),x.shape[0])]
    return solxx



x = np.array([1,-1,-1,-1,1,0,1,2])
y = np.array([5,-4,3,2,-1,1,0,-1])
print cyc_cunv(x,y)
f_x = dtft(x)
f_y = dtft(y)
x_dot_y = f_x * f_y
print x_dot_y
print invers_dtft(x_dot_y)