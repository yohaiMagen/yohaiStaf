import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.offsetbox as offsetbox




input1= pd.read_excel("X:\\Nicole_n_Gil\\SinkHoles_Data\\intervals_DATA.xlsx", sheetname=2)
data = input1.as_matrix()
data = data[:,[13,17]]


#print(data)
np.random.shuffle(data)
#print("__________________________________")
#print(data)
#print(data.shape)

interval = data[:,0]
lit_vec = data[:,1]
#print(interval)
#print(lit_vec)
def expfit(x, a, b, c):
	return  np.exp(b*x)*a + c
	
def linfit(x, a, b):
	return a*x + b
	
def R_2(X, Y, func, popt):
	resiudal = Y - func(X, *popt)
	ss_res = np.sum(resiudal ** 2)
	ss_tot = np.sum((Y - np.mean(Y)) ** 2)
	return 1 - (ss_res / ss_tot)
	


start = 0
end = 6

exp_loss = 0
x = np.linspace(0,1,100)
for i in range(0,6):
	temp_lit = np.concatenate((lit_vec[0:start], lit_vec[end:35]))
	temp_interval = np.concatenate((interval[0:start], interval[end:35]))
	temp_lit = np.asfarray(temp_lit)
	temp_interval = np.asfarray(temp_interval)
	popt, pcov = curve_fit(expfit, temp_lit, temp_interval)
	fitlabel = '${0:.3f}\cdot EXP({1:.3f}\cdot x ) + {2:.3f}$'.format(popt[0], popt[1], popt[2])
	plt.figure()
	plt.plot(x, expfit(x, *popt), color='b', label=fitlabel)
	plt.plot(temp_lit, temp_interval, 'ko', color='c', label='train set')
	plt.plot(lit_vec[start:end], interval[start:end], 'ko', color='r', label='validation set')
	val_lit = np.asfarray(lit_vec[start:end])
	val_interval = np.asfarray(interval[start:end])
	resiudal = val_interval - expfit(val_lit, *popt)
	r_2 = np.mean(resiudal**2)
	exp_loss += r_2
	#ss_res = np.sum(resiudal ** 2)
	#ss_tot = np.sum((val_interval - np.mean(val_interval)) ** 2)
	#print("____________")
	#print(ss_res / ss_tot)
	#print("____________")
	#r_2 =  1 - (ss_res / ss_tot)
	leg = plt.legend()
	leg_r_squerd = "r^2 = {0:.3f}".format(r_2)
	txt=offsetbox.TextArea(leg_r_squerd)
	box = leg._legend_box
	box.get_children().append(txt)
	box.set_figure(box.figure)
	start = start + 6
	end = end + 6

exp_loss = exp_loss/6

start = 0
end = 6
lin_loss = 0
for i in range(0,6):
	temp_lit = np.concatenate((lit_vec[0:start], lit_vec[end:35]))
	temp_interval = np.concatenate((interval[0:start], interval[end:35]))
	temp_lit = np.asfarray(temp_lit)
	temp_interval = np.asfarray(temp_interval)
	popt, pcov = curve_fit(linfit, temp_lit, temp_interval)
	fitlabel = '${0:.3f}\cdot x  + {1:.3f}$'.format(popt[0], popt[1])
	plt.figure()
	plt.plot(x, linfit(x, *popt), color='b', label=fitlabel)
	plt.plot(temp_lit, temp_interval, 'ko', color='c', label='train set')
	plt.plot(lit_vec[start:end], interval[start:end], 'ko', color='r', label='validation set')
	val_lit = np.asfarray(lit_vec[start:end])
	val_interval = np.asfarray(interval[start:end])
	resiudal = val_interval - linfit(val_lit, *popt)
	r_2 = np.mean(resiudal**2)
	lin_loss += r_2
	#ss_res = np.sum(resiudal ** 2)
	#ss_tot = np.sum((val_interval - np.mean(val_interval)) ** 2)
	#print("____________")
	#print(ss_res / ss_tot)
	#print("____________")
	#r_2 =  1 - (ss_res / ss_tot)
	leg = plt.legend()
	leg_r_squerd = "r^2 = {0:.3f}".format(r_2)
	txt=offsetbox.TextArea(leg_r_squerd)
	box = leg._legend_box
	box.get_children().append(txt)
	box.set_figure(box.figure)
	start = start + 6
	end = end + 6
	
lin_loss = lin_loss/6

start = 0
end = 6
poly_loss = 0
for i in range(0,6):
	temp_lit = np.concatenate((lit_vec[0:start], lit_vec[end:35]))
	temp_interval = np.concatenate((interval[0:start], interval[end:35]))
	temp_lit = np.asfarray(temp_lit)
	temp_interval = np.asfarray(temp_interval)
	popt = np.polyfit(temp_lit, temp_interval, 2)
	fitlabel = '${0:.3f}\cdot x^2  + {1:.3f} \cdot x + {2:.3f}$'.format(popt[0], popt[1], popt[2])
	plt.figure()
	p = np.poly1d(popt)
	plt.plot(x, p(x), color='b', label=fitlabel)
	plt.plot(temp_lit, temp_interval, 'ko', color='c', label='train set')
	plt.plot(lit_vec[start:end], interval[start:end], 'ko', color='r', label='validation set')
	val_lit = np.asfarray(lit_vec[start:end])
	val_interval = np.asfarray(interval[start:end])
	resiudal = val_interval - p(val_lit)
	r_2 = np.mean(resiudal**2)
	poly_loss += r_2
	#ss_res = np.sum(resiudal ** 2)
	#ss_tot = np.sum((val_interval - np.mean(val_interval)) ** 2)
	#print("____________")
	#print(ss_res / ss_tot)
	#print("____________")
	#r_2 =  1 - (ss_res / ss_tot)
	leg = plt.legend()
	leg_r_squerd = "r^2 = {0:.3f}".format(r_2)
	txt=offsetbox.TextArea(leg_r_squerd)
	box = leg._legend_box
	box.get_children().append(txt)
	box.set_figure(box.figure)
	start = start + 6
	end = end + 6
	
poly_loss = poly_loss/6

start = 0
end = 6
poly_loss_3 = 0
for i in range(0,6):
	temp_lit = np.concatenate((lit_vec[0:start], lit_vec[end:35]))
	temp_interval = np.concatenate((interval[0:start], interval[end:35]))
	temp_lit = np.asfarray(temp_lit)
	temp_interval = np.asfarray(temp_interval)
	popt = np.polyfit(temp_lit, temp_interval, 3)
	fitlabel = '${0:.3f}\cdot x^3  + {1:.3f} \cdot x^2  + {2:.3f} \cdot x + {3:.3f}$'.format(popt[0], popt[1], popt[2], popt[3])
	plt.figure()
	p = np.poly1d(popt)
	plt.plot(x, p(x), color='b', label=fitlabel)
	plt.plot(temp_lit, temp_interval, 'ko', color='c', label='train set')
	plt.plot(lit_vec[start:end], interval[start:end], 'ko', color='r', label='validation set')
	val_lit = np.asfarray(lit_vec[start:end])
	val_interval = np.asfarray(interval[start:end])
	resiudal = val_interval - p(val_lit)
	r_2 = np.mean(resiudal**2)
	poly_loss_3 += r_2
	#ss_res = np.sum(resiudal ** 2)
	#ss_tot = np.sum((val_interval - np.mean(val_interval)) ** 2)
	#print("____________")
	#print(ss_res / ss_tot)
	#print("____________")
	#r_2 =  1 - (ss_res / ss_tot)
	leg = plt.legend()
	leg_r_squerd = "r^2 = {0:.3f}".format(r_2)
	txt=offsetbox.TextArea(leg_r_squerd)
	box = leg._legend_box
	box.get_children().append(txt)
	box.set_figure(box.figure)
	start = start + 6
	end = end + 6
	
poly_loss_3 = poly_loss_3/6

print(np.sqrt(exp_loss))
print(np.sqrt(lin_loss))
print(np.sqrt(poly_loss))
print(np.sqrt(poly_loss_3))

	
plt.show()
