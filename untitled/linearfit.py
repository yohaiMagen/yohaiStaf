import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.offsetbox as offsetbox



input1= pd.read_excel('/home/yohai/Downloads/intervals_DATA.xlsx', sheetname=2)
data = input1.as_matrix()


lit_vec = data[:,[17, 19]]
depth_vec = data[:,14]
thikness_vec = data[:,16]
interval = data[:,13]
lit_vec = np.asfarray(lit_vec)
depth_vec = np.asfarray(depth_vec)
thikness_vec = np.asfarray(thikness_vec)
interval = np.asfarray(interval)
depth_vec = depth_vec/np.max(depth_vec)
# print(lit_vec)
con = lit_vec[:,1] < 10
con = np.array([con]).T
# print(con)
lit_vec = np.extract(con, lit_vec[:,0])
depth_vec =np.extract(con, depth_vec)
interval = np.extract(con, interval)
print(thikness_vec)
lit_vec = lit_vec/np.max(lit_vec)
thikness_vec = thikness_vec/np.max(thikness_vec)




def func(x, a, b):
    return a*x + b;

lit = 0
best_lit = 0
depth = 1
best_depth = 1
max_r_squared = 0
best_popt = 0
while depth >= 0:
    combine = lit * lit_vec + depth * depth_vec
    popt, pcov = curve_fit(func, combine, interval)
    resiudal = interval - func(combine, *popt)
    ss_res = np.sum(resiudal ** 2)
    ss_tot = np.sum((interval - np.mean(interval)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    if r_squared > max_r_squared:
        max_r_squared = r_squared
        best_lit = lit
        best_depth = depth
        best_popt = popt
    lit += 0.001
    depth -= 0.001



# plt.rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

popt1, pcov1 = curve_fit(func, lit_vec, interval)
fitlabel = '${0:.3f}*x + {1:.3f}$'.format(popt1[0], popt1[1])
ax1.plot(lit_vec , func(lit_vec , *popt1), 'r-', label=fitlabel)
ax1.plot(lit_vec, interval, 'ko', color='c', label='lithology to time interval')
resiudal = interval - func(lit_vec, *popt1)
ss_res = np.sum(resiudal ** 2)
ss_tot = np.sum((interval - np.mean(interval)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
leg = ax1.legend( loc=[0, .995])
leg_r_squerd = "r^2 = {0:.3f}".format(r_squared)
txt=offsetbox.TextArea(leg_r_squerd)
box = leg._legend_box
box.get_children().append(txt)
box.set_figure(box.figure)
ax1.set_xlim(0, 1.1)
ax1.set_xlabel('$\\frac{gravel\:percentage}{gravel\:percentage\:maximum}$', fontsize = 25)
ax1.set_ylabel('days', fontsize = 22)



popt1, pcov1 = curve_fit(func, thikness_vec, interval)
fitlabel = '${0:.3f}*x + {1:.3f}$'.format(popt1[0], popt1[1])
ax2.plot(thikness_vec , func(thikness_vec , *popt1), 'r-', label=fitlabel)
ax2.plot(thikness_vec, interval, 'ko', color='g', label='halite thikness to time interval')
resiudal = interval - func(thikness_vec, *popt1)
ss_res = np.sum(resiudal ** 2)
ss_tot = np.sum((interval - np.mean(interval)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
leg = ax2.legend( loc=[0, .995])
leg_r_squerd = "r^2 = {0:.3f}".format(r_squared)
txt=offsetbox.TextArea(leg_r_squerd)
box = leg._legend_box
box.get_children().append(txt)
box.set_figure(box.figure)
ax2.set_xlim(0, 1.1)
ax2.set_xlabel('$\\frac{halite\:thikness}{halite\:thikness\:maximum}$', fontsize = 25)



popt2, pcov2 = curve_fit(func, depth_vec, interval)
fitlabel = '${0:.3f}*x \:{1:.3f}$'.format(popt2[0], popt2[1])
ax3.plot(depth_vec , func(depth_vec , *popt2), 'r-', label=fitlabel)
ax3.plot(depth_vec, interval, 'ko', color='r', label='halite depth to time interval')

resiudal = interval - func(depth_vec, *popt2)
ss_res = np.sum(resiudal ** 2)
ss_tot = np.sum((interval - np.mean(interval)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

leg = ax3.legend( loc=[0, .995])
leg_r_squerd = "r^2 = {0:.3f}".format(r_squared)
txt=offsetbox.TextArea(leg_r_squerd)
box = leg._legend_box
box.get_children().append(txt)
box.set_figure(box.figure)
ax3.set_xlim(0, 1.1)
ax3.set_xlabel('$\\frac{halite\:depth}{halite\:depth\: maximum}$', fontsize = 25)
plt.savefig('/home/yohai/Documents/סמינריון/fig1.jpg')


# plt.figure()
# combine = "${0:.3f}* lithology +{1:.3f} * halite\: depth$\n$ to\: time\: interval$".format(best_lit, best_depth)
# data = plt.plot(best_depth*depth_vec + best_lit*lit_vec , interval, 'ko', label=combine)
# fitlabel = '${0:.3f}*x \: {1:.3f}$'.format(best_popt[0], best_popt[1])
# fit = plt.plot(best_depth*depth_vec + best_lit*lit_vec , func(best_depth*depth_vec + best_lit*lit_vec , *best_popt), 'r-', label=fitlabel)
# ax = plt.gca()
#
#
# leg = plt.legend( loc='upper left')
# leg_r_squerd = "r^2 = {0:.3f}".format(max_r_squared)
# txt=offsetbox.TextArea(leg_r_squerd)
# box = leg._legend_box
# box.get_children().append(txt)
# box.set_figure(box.figure)
# plt.xlim(0, 1.1)



plt.show()
