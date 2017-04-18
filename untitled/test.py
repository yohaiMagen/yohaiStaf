import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

a =  np.loadtxt(open("/home/yohai/Documents/all_profiles1.csv", "r"), delimiter= ",", skiprows=3)
print(a[1])

def func(x, a, b):
    return a *(1-np.exp(-b * x))



popt, pcov = curve_fit(func, a[5], a[1])
plt.figure()
data = plt.plot(a[5], a[1], 'ko', label="Original Noised Data")
fit = plt.plot(a[5], func(a[5], *popt), 'r-', label="Fitted Curve")
ax = plt.gca()
vc = 5
str =  "$a*(1-e^{-bx})$\na = %f , b = %f" %(popt[0], popt[1])
plt.figtext(.27,.63, str)
plt.legend( bbox_to_anchor=(0.6, 0.9), bbox_transform=ax.transAxes)
print(popt)
print(pcov)
plt.show()