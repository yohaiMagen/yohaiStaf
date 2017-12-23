import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.offsetbox as offsetbox



input1= pd.read_excel('/home/yohai/Downloads/geoData.xlsx', sheetname='data')
data = input1.as_matrix()
print(data)


x = np.asfarray(data[1:,2])
print(x.shape)
t = np.asfarray(data[1:,3])
print(t.shape)
x_2 = x**2
t_2 = t**2

p = np.polyfit(x_2, t_2, 1)

func = np.poly1d(p)




lab = "$t^2 = {0:.7f} \cdot x^2 + {1:.7f}$ ".format(p[0], p[1])
X = np.linspace(0,1600000,1600000)
plt.plot(X, func(X), label=lab)
plt.plot(x_2, t_2, 'ko', color='r')
plt.xlabel("$x^2 [m^2]$")
plt.ylabel("$t^2 [s^2]$")
plt.legend(loc=[0, .9])
plt.title("wave 2")
plt.show()

