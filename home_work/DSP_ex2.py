import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as ptc

def func(omega, N):
    h_of_omega = 0
    for i  in range(1, N+1):
        h_of_omega += np.sin(np.pi / 6 * i) / (np.pi * i) *np.exp(-omega*i*1j)*2
    return h_of_omega + 1.0/6


x = np.linspace(-np.pi/ 3, np.pi / 3, 200)
fig, ax = plt.subplots()

for n, c in zip([2, 5, 15, 30], ['r', 'b', 'g', 'c']):
    ax.plot(x, np.real(func(x, n)), color=c, label=n)

ax.add_patch(ptc.Rectangle((-np.pi/6, 0), np.pi/3, 1, fc='none', ec='k', label='filter'))
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.3))
plt.legend()
plt.show()