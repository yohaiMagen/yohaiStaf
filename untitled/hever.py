import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.offsetbox as offsetbox




input1= pd.read_excel("X:\\Nicole_n_Gil\\yohai\\hever\\profiles\\profiles.xlsx", sheetname='Sheet2')
data = input1.as_matrix()
sub_from_begining = data[3,1:]
days_from_begining_to_end = data[8,1:]
sub_from_begining = np.asfarray(sub_from_begining)
days_from_begining_to_end = np.asfarray(days_from_begining_to_end)



def expfit(x, a, b):
	return a * np.exp(b*x)
	
fig = plt.figure()
ax1 = fig.add_subplot(111)
popt1, pcov1 = curve_fit(expfit, days_from_begining_to_end, sub_from_begining, bounds=([0, 0], [0.1, 0.1]))
print(popt1)
fitlabel = '${0:.3f}\cdot EXP({1:.3f}\cdot x )$'.format(popt1[0], popt1[1])
xf = np.linspace(0, 1750, 500)
ax1.plot(xf , expfit(xf , *popt1), 'k', label=fitlabel)
ax1.plot(days_from_begining_to_end, sub_from_begining, 'ko', color='k', label='total subsidence', ms=2.5)
resiudal = sub_from_begining - expfit(days_from_begining_to_end, *popt1)
ss_res = np.sum(resiudal ** 2)
ss_tot = np.sum((sub_from_begining - np.mean(sub_from_begining)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
leg = ax1.legend( loc=[0.05, .2])
leg_r_squerd = "r^2 = {0:.3f}".format(r_squared)
txt=offsetbox.TextArea(leg_r_squerd)
box = leg._legend_box
box.get_children().append(txt)
box.set_figure(box.figure)
#ax1.set_xlim(0, 1.1)
ax1.set_xlabel('Time since 12/30/2011(days)', fontsize = 16)
ax1.set_ylabel('cumulative Displacement(cm)', fontsize = 16)
plt.gca().invert_yaxis()
plt.show()
