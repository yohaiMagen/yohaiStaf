import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

x = np.array([0, 10, 20, 30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240, 250])
forward = np.array([0,10,20,30,40,50,60,70,80,90,99,106,112,119,125,131,138,144,151,157,164,170,176,183,189,195])
revers = np.array([195,191,188,184,181,177,174, 171,167,160,150, 140, 130, 120,110, 100, 90,80, 70, 60, 50, 34,30,20, 10, 0])

x_1 = np.array([0, 10, 20, 30,40,50,60,70,80,90])
x_2 = np.array([100,110,120,130,140,150,160,170,180,190,200,210,220,230,240, 250])
revers_1 = np.array([195,191,188,184,181,177,174, 171,167,160])
revers_2 = np.array([150, 140, 130, 120,110, 100, 90,80, 70, 60, 50, 34,30,20, 10, 0])
x_1f = np.array([0, 10, 20, 30,40,50,60,70,80])
x_2f = np.array([90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240, 250])
forward_1 = np.array([0,10,20,30,40,50,60,70,80])
forward_2 = np.array([90,99,106,112,119,125,131,138,144,151,157,164,170,176,183,189,195])
xf = np.linspace(0,250,250)


def func(x ,a, b):
    return a*x + b


popt1, pcov1 = curve_fit(func, x_1, revers_1)
popt2, pcov2 = curve_fit(func, x_2, revers_2)
fitlabel_1 = 'revers v1 = ${0:.3f}\cdot x + {1:.3f}$'.format(popt1[0], popt1[1])
fitlabel_2 = 'revers v2 = ${0:.3f}\cdot x + {1:.3f}$'.format(popt2[0], popt2[1])

print(xf)
print(func(xf, *popt1))
plt.plot(x, revers, 'ko',  color='r')
plt.plot(x, forward, 'ko', color='b')
plt.plot(xf, func(xf, *popt1), color='r', label=fitlabel_1)
plt.plot(xf, func(xf, *popt2), color='r', label=fitlabel_2)

popt1, pcov1 = curve_fit(func, x_1f, forward_1)
popt2, pcov2 = curve_fit(func, x_2f, forward_2)

fitlabel_3 = 'forward v1 = ${0:.3f}\cdot x + {1:.3f}$'.format(popt1[0], popt1[1])
fitlabel_4 = 'forward v2 = ${0:.3f}\cdot x + {1:.3f}$'.format(popt2[0], popt2[1])

plt.plot(xf, func(xf, *popt1), color='b', label=fitlabel_4)
plt.plot(xf, func(xf, *popt2), color='b', label=fitlabel_3)
plt.legend(loc=[0,.72])
plt.show()







