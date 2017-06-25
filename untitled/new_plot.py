import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.offsetbox as offsetbox

def expfit(x, a, b, c):
	return a * np.exp(b*x) + c

x1 = pd.ExcelFile("X:\\Nicole_n_Gil\\SinkHoles_Data\\opening intervals\\intervals_DATA.xlsx")
print(x1.sheet_names)

print('fiting data')
input1= pd.read_excel("X:\\Nicole_n_Gil\\SinkHoles_Data\\opening intervals\\intervals_DATA.xlsx", sheetname="filterd_data")
#print(input1)
data = input1.as_matrix()


lit_vec = data[:,17]
depth_vec = data[:,14]

thikness_vec = data[:,16]
interval = data[:,13]
xf = np.linspace(0,1.1,100)

print(lit_vec)
lit_vec = np.asfarray(lit_vec)
depth_vec = np.asfarray(depth_vec)

thikness_vec = np.asfarray(thikness_vec)
interval = np.asfarray(interval)
depth_vec = depth_vec/np.max(depth_vec)

ax1 = plt.figure().add_subplot(111)
ax2 = plt.figure().add_subplot(111)

popt1, pcov1 = curve_fit(expfit, lit_vec, interval)
fitlabel = '${0:.3f}\cdot EXP({1:.3f}\cdot x ) + {2:.3f}$'.format(popt1[0], popt1[1], popt1[2])
ax1.plot(xf , expfit(xf , *popt1), 'k', label=fitlabel)
ax1.plot(xf , expfit(xf , *popt1) + 41.75, 'r--', label='confidence')
ax1.plot(xf , expfit(xf , *popt1) - 41.75, 'r--')
ax1.plot(lit_vec, interval, 'ko', color='b', label='lithology to time interval')
ax1.errorbar(lit_vec, interval, fmt='o', yerr=16, color='b')
resiudal = interval - expfit(lit_vec, *popt1)
ss_res = np.sum(resiudal ** 2)
ss_tot = np.sum((interval - np.mean(interval)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
leg = ax1.legend( loc=[0.05, .8])
leg_r_squerd = "r^2 = {0:.3f}".format(r_squared)
txt=offsetbox.TextArea(leg_r_squerd)
box = leg._legend_box
box.get_children().append(txt)
box.set_figure(box.figure)
ax1.set_xlim(0, 1.1)
ax1.set_xlabel('$\\frac{gravel\:percentage}{gravel\:percentage\:maximum}$', fontsize = 25)
ax1.set_ylabel('days', fontsize = 22)



ax2.semilogy(xf , expfit(xf , *popt1), 'k', label=fitlabel)
ax2.semilogy(xf , expfit(xf , *popt1) + 41.75, 'r--', label='confidence')
ax2.semilogy(xf , expfit(xf , *popt1) - 41.75, 'r--')
ax2.semilogy(lit_vec, interval, 'ko', color='b', label='lithology to time interval')
ax2.errorbar(lit_vec, interval, fmt='o', yerr=16, color='b')
leg = ax2.legend( loc=[0.05, .8])
leg_r_squerd = "r^2 = {0:.3f}".format(r_squared)
txt=offsetbox.TextArea(leg_r_squerd)
box = leg._legend_box
box.get_children().append(txt)
box.set_figure(box.figure)
ax2.set_xlim(0, 1.1)
ax2.set_xlabel('$\\frac{gravel\:percentage}{gravel\:percentage\:maximum}$', fontsize = 25)
ax2.set_ylabel('days', fontsize = 22)
plt.show()
