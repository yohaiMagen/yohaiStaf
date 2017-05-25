import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


X = np.load('/home/yohai/Documents/IML/Ex4/X_poly.npy')
Y = np.load('/home/yohai/Documents/IML/Ex4/Y_poly.npy')

X_train = X[0:99]
X_val = X[100:199]
X_test = X[200:299]

Y_train = Y[0:99]
Y_val = Y[100:199]
Y_test = Y[200:299]
D = [None] * 16


def R_squerd(func, x, y):
    resiudal = y - func(x)
    ss_res = np.sum(resiudal ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)


popt = [None]*16
p = [None]*16

for i in range(1,16):
    popt[i] = np.polyfit(X_train, Y_train, i)
    D[i] = np.poly1d(popt[i])


h_star = 0
best_r_squerd = 0
for i in range(1,16):
    r_squerd = R_squerd(D[i], X_val, Y_val)
    if r_squerd > best_r_squerd:
        best_r_squerd = r_squerd
        h_star = i

train_err = np.array([])
val_err = np.array([])
for i in range(1,16):
    train_err = np.append(train_err, np.array([R_squerd(D[i], X_train, Y_train)]))
    val_err = np.append(val_err, [R_squerd(D[i], X_val, Y_val)])

a = np.linspace(1,15,15)
f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
ax2.plot(a, train_err, color='g', label='train error')
ax2.plot(a, train_err,'ko', color='g')
ax2.plot(a, val_err, color='b', label='validation error')
ax2.plot(a, val_err, 'ko', color='b')
ax.plot(a, train_err, color='g', label='train error')
ax.plot(a, val_err, color='b', label='validation error')
ax.plot(a, val_err, 'ko', color='b')
ax.plot(a, train_err,'ko', color='g')
ax2.set_ylim(0.941, 0.9415)
ax.set_ylim(0.9962,.9978)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
plt.legend()

plt.figure(2)
plt.plot(X_val, Y_val, 'ko')
x= np.linspace(0, 1, 200)
plt.plot(x, D[h_star](x), color='r', label='best fit is polynom of {0:d} deg'.format(h_star))


r_squerd_fold = 0
h_fold = 0
for i in range(1,16):
    start = 0
    end = 39
    r_squerd_fold_temp = 0
    for j in range(1,5):
        pv = np.polyfit(np.concatenate((X[0:start], X[end:199])), np.concatenate((Y[0:start], Y[end:199])), i)
        p = np.poly1d(pv)
        r_squerd_fold_temp = r_squerd_fold_temp +  R_squerd(p, X[start:end], Y[start:end])
        start += 40
        end += 40
    r_squerd_fold_temp = r_squerd_fold_temp/5
    if r_squerd_fold_temp > r_squerd_fold:
        h_fold = i
        r_squerd_fold = r_squerd_fold_temp

plt.plot(x, D[h_fold](x), color='g', label='best fit is of fold alg is {0:d} deg'.format(h_star))
plt.legend()
plt.show()