import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import matplotlib.cm as cm

X, Y = np.meshgrid(np.arange(-15, 15, 1), np.arange(-15, 15, 1))
U = -Y+2*X
V = -X

plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, U, V, units='width')
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.xlim(-10,10)
plt.ylim(-10,10)

plt.figure()
n = 256
x = np.arange(-15., 15., 1)
y = np.arange(-15., 15., 1)
X, Y = np.meshgrid(x, y)
Z = -X*Y +X**2

plt.pcolormesh(X, Y, Z, cmap = cm.gray)
plt.figure()
U, V = np.gradient(Z)
Q = plt.quiver(X, Y, V, U, units='width')
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
# plt.xlim(0, 30)
# plt.ylim(0, 30)
plt.show()