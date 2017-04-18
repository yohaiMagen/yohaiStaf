import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import pandas as pd



input1= pd.read_excel('/home/yohai/Downloads/intervals_DATA.xlsx', sheet_name='Sheet3', skiprows=(1))
data = input1.as_matrix()


lit_vec = data[:,16]
depth_vec = data[:,14]
some_other_vec = data[:,15]
interval = data[:,13]
lit_vec = np.asfarray(lit_vec)
depth_vec = np.asfarray(depth_vec)
some_other_vec = np.asfarray(some_other_vec)
interval = np.asfarray(interval)
print(interval)

def func(x, a, b):
    return a*x+b;

lit = 1
best_lit = 1
depth = 0
best_depth = 0
some_other = 0
best_some_other = 0
max_r_squared = 0
best_popt = 0
while lit >= 0:
    some_other = 1-lit
    depth = 0
    while some_other >= 0:
        combine = lit * lit_vec + depth * depth_vec + some_other*some_other_vec
        popt, pcov = curve_fit(func, combine, interval)
        resiudal = interval - func(combine, *popt)
        ss_res = np.sum(resiudal ** 2)
        ss_tot = np.sum((interval - np.mean(interval)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > max_r_squared:
            max_r_squared = r_squared
            best_lit = lit
            best_depth = depth
            best_some_other = some_other
            best_popt = popt
        some_other -= 0.005
        depth += 0.005
    lit -= 0.005
    print(lit)






plt.figure()
plt.plot(lit_vec, interval, 'ko', color='c')
plt.plot(depth_vec, interval, 'ko', color='r')
plt.plot(some_other_vec, interval, 'ko', color='g')
data = plt.plot(best_depth*depth_vec + best_lit*lit_vec + best_some_other*some_other_vec, interval, 'ko', label="Original Noised Data")
fit = plt.plot(best_depth*depth_vec + best_lit*lit_vec + best_some_other*some_other, func(best_depth*depth_vec + best_lit*lit_vec +best_some_other*some_other, *best_popt), 'r-', label="Fitted Curve")
ax = plt.gca()
print(best_popt)
print(best_lit)
print(best_depth)
print(best_some_other)
print(max_r_squared)
plt.show()
