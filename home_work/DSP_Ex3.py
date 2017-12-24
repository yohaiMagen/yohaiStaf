import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as ptc
import numpy as np
import time


def Q1():
    a1 = ptc.Rectangle((80*np.pi, 0),1,np.pi,color='r')
    a2 = ptc.Rectangle((45*np.pi, 0),1,np.pi,color='r')
    a3 = ptc.Rectangle((37.5*2*np.pi, 0),1,np.pi,color='r')
    fig = plt.figure()
    ax = plt.gca()
    ax.add_patch(a1)
    ax.add_patch(a2)
    ax.add_patch(a3)
    ax.set_xlim(0, 100*np.pi)
    ax.set_ylim(0, 1.5*np.pi)
    plt.show()

def sinc(x):
    return np.sin(x)/x

def Q2():
    t = np.arange(-0.5, 0.5, 0.008)
    sinct = sinc(t/0.008)
    plt.plot(t, sinct)
    plt.show()


def f(t):
    return np.exp(-t**2)


def sinc_interpolation(t, x, N, dt,  zero = None):

    if zero is None:
        zero = int(len(x)/2)
    start = int(np.floor(t))- N
    res = 0
    for n in range(start, start+2*N +1, 1):
        res += x[zero+n]*sinc(np.pi*(t-n*dt)/dt)
    return res

def vsinc_interpolation(t, x, N, dt):
    sol = np.zeros(len(t))
    for i in range(len(t)):
        sol[i] = sinc_interpolation(t[i], x, N, dt)
    return sol


def Q3():
    x = np.arange(-1000, 1000, 0.5)
    t = np.arange(-10, 10, 0.05)
    fig = plt.figure(figsize=(20,10))
    ax1, ax2, ax3 = fig.subplots(3)
    l1, = ax1.plot(t, vsinc_interpolation(t, f(x), 2, 0.5), color='b', label='N = 5')
    l2, = ax2.plot(t, vsinc_interpolation(t, f(x), 3, 0.5), color='g', label='N = 7')
    l3, = ax3.plot(t, vsinc_interpolation(t, f(x), 4, 0.5), color='c', label='N = 9')
    l4, = ax1.plot(t, f(t), color='r', linewidth=0.8, label='$e^{-t^{2}}$')
    ax2.plot(t, f(t), color='r', linewidth=0.8)
    ax3.plot(t, f(t), color='r', linewidth=0.8)
    fig.legend((l1, l2, l3, l4), ('N = 5', 'N = 7', 'N = 9', '$e^{-t^{2}}$'), 'upper left')
    plt.show()


Q2()
