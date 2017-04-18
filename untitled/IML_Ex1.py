import numpy as np
import matplotlib.pylab as plt


data = np.random.binomial(1, 0.25, (1000, 1000))
epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
m = np.arange(1, 1001, 1)


def mean(n):
    a = np.array([0])
    for i in range(1, 1000, 1):
        c = np.array([np.mean(data[n][0:i])])
        a = np.concatenate([a, c])
    return a


def chebyshev(n, eps):
    return 1/(4*m*np.power(eps, 2))


def hoeffding(n, eps):
    return 2*np.exp(-2*n*np.power(eps, 2))


def sequwncesPrec(n, eps):
    prec = []
    for i in range(1, n+1):
        mean_i = np.mean(data[:, 0:i], axis=1)
        sum = np.sum(np.abs(mean_i - 0.25) >= eps)
        prec.append(float(sum) / data.shape[0])
    return prec




def q1():
    b1 = mean(0)
    b2 = mean(1)
    b3 = mean(2)
    b4 = mean(3)
    b5 = mean(4)
    plt.figure(0)
    plt.plot(m, b1, color='r', label='row number 1')
    plt.plot(m, b2, color='y', label='row number 2')
    plt.plot(m, b3, color='c', label='row number 3')
    plt.plot(m, b4, color='k', label='row number 4')
    plt.plot(m, b5, color='g', label='row number 5')
    plt.legend()

def q2A(eps):
    prec = sequwncesPrec(1000, eps)
    plt.plot(m, prec, color='c', linewidth=2, label="percent of rows that satisfy $\\vert \\bar X_m - E(X) \\vert \\geq \\epsilon$")
    a = chebyshev(m,eps)
    plt.plot(m, chebyshev(m, eps), linewidth=2, label='chebyshev')
    plt.plot(m, hoeffding(m, eps), color = 'r', linewidth=2, label='hoeffding')
    plt.title("$\\varepsilon = $%f" %(eps))


def q2():
    plt.figure(1)
    q2A(epsilon[0])
    plt.ylim(0, 0.3)
    plt.legend()
    plt.figure(2)
    q2A(epsilon[1])
    plt.ylim(0, 0.3)
    plt.legend()
    plt.figure(3)
    q2A(epsilon[2])
    plt.ylim(0, 0.8)
    plt.legend()
    plt.figure(4)
    q2A(epsilon[3])
    plt.ylim(0, 1)
    plt.legend()
    plt.figure(5)
    q2A(epsilon[4])
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(0.9, 0.5))


q1()
q2()
plt.show()


