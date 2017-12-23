from apsg import *
from windrose_edit import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

fw = pd.read_csv('/home/yohai/Documents/structural_report/faults.csv')
print fw
mat  = fw.as_matrix()
# print mat[:,6]
# print mat[0,0]
# print mat[0,1]
# print mat.shape
# print mat

#
#
#
# faults1 = []
# faults2 = []
# rake = []
# s1 = StereoNet(grid=True)
# s2 = StereoNet(grid=True)
# for i in range(mat.shape[0]):
#      if mat[i,3] == 1:
#          faults1.append(Fol(mat[i,1], mat[i,0]))
#      elif mat[i,3] == 2:
#          faults2.append(Fol(mat[i,1], mat[i,0]))
#      else:
#          s2.plane(Fol(mat[i,1] , mat[i,0]))
#
#      # rake .append(faults[i].rake(mat[i,2]).aslin)
# f1 = Group(faults1, 'faults group 1')
# f2 = Group(faults2, 'faults group 2')
# # r = Group(rake, 'Rake')
#
#
# s1.plane(f1, color='b')
# s1.plane(f2, color='g')
# # s.line(r, label='Rake', color='g')
#
#
#
# # print mat[:,1]
ax = WindroseAxes.from_ax()
ax.bar(np.asfarray(np.mod(mat[:,1]-90,360)), mat[:,0],  opening=0.8, edgecolor='white')
ax.legend(loc=(0.8,0), title="Dip")


fualts_1 =np.asfarray(mat[mat[:,3] == 1])
fualts_2 = np.asfarray(mat[mat[:,3] == 2])
s_1 = StereoNet(grid=True)
# print fualts_1[:,1]
# print fualts_2[:,1]
a = np.mean(fualts_2[:,0])
g1 = Fol(np.mean(fualts_1[:,1]), np.mean([fualts_1[:,0]]))
g2 = Fol(np.mean(fualts_2[:,1]), np.mean([fualts_2[:,0]]))
g1_rake = Fol(np.mean(fualts_1[:,1]), np.mean([fualts_1[:,0]])).rake(np.mean(180 - fualts_1[:,2])).aslin
g2_rake = Fol(np.mean(fualts_2[:,1]), np.mean([fualts_2[:,0]])).rake(np.mean(fualts_2[:,2])).aslin
dip_1 = np.mean(fualts_1[:,0])
# dire_1 = np.mean(fualts_1[:,1])
# rake_1 = np.mean(180+fualts_1[:,2])
# dip_2 = np.mean(fualts_2[:,0])
# dire_2 = np.mean(fualts_2[:,1])
# rake_2 = np.mean(fualts_2[:,2])
# trend_1 =  np.mod(dire_1 -90 + np.rad2deg(np.arctan(np.tan(np.deg2rad(rake_1))*np.cos(np.deg2rad(dip_1)))),360)
# trend_2 =  np.mod(dire_2 -90 + np.rad2deg(np.arctan(np.tan(np.deg2rad(rake_2))*np.cos(np.deg2rad(dip_2)))),360)
# s_1.plane(Fol((trend_1+trend_2/2-90), 90))
# s_1.plane((Fol(125,90)))
# print trend_2

# s_1.plane(g1, color = 'b')
# s_1.plane(g2, color = 'g')
# s_1.line(g1_rake, color = 'r')
# s_1.line(g2_rake, color = 'r')

s_1.plane(g1.rotate(Lin(0,0), 27), color = 'b')
s_1.plane(g2.rotate(Lin(0,0), 27), color = 'g')
s_1.line(g1_rake.rotate(Lin(0,0), 27), color = 'r')
s_1.line(g2_rake.rotate(Lin(0,0), 27), color = 'r')
# rake .append(faults[i].rake(mat[i,2]).aslin)
plt.show()

