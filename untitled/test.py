import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

a = np.array(['democrat.', 'democrat.', 'democrat.', 'democrat.', 'democrat.', 'democrat.', 'republican.', 'republican.', 'republican.', 'republican.'])
print(a)
b = a == 'republican.'
print(b)