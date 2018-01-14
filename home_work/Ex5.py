import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def real(omega):
    return 1-1.25*np.cos(omega-(2/3)*np.pi)

def im(omega):
    return 1.25*np.sin(omega-(2/3)*np.pi)

omega = np.linspace(-2*np.pi, 2*np.pi, 1000)
f,ax=plt.subplots(1)
x=np.linspace(0, 3*np.pi, 1001)
ax.plot(omega, real(omega), color='r', label='real')
ax.plot(omega, im(omega), color='b', label='imaginary')
ax.plot(omega, np.arctan(im(omega)/real(omega)), color='g', label='phase')
ax.set_xticks([-2*np.pi, -1.5*np.pi, -np.pi, -0.5*np.pi, 0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
ax.set_xticklabels([r"$-2\pi$", r"$-\frac{3}{2}\pi$", r"$-\pi$", r"$-\frac{1}{2}\pi$", "$0$", r"$\frac{1}{2}\pi$",
                    r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])
plt.legend()
plt.show()