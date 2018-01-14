import numpy as np
import matplotlib.pyplot as plt

A = np.fromfile('A.txt', sep='\n')
B = np.fromfile('B.txt', sep='\n')
w1 = np.fromfile('WAVELT1.txt', sep='\n')
w2 = np.fromfile('WAVELT2.txt', sep='\n')

A1 = np.convolve(A, w1)
A2 = np.convolve(A, w2)
B1 = np.convolve(B, w1)
B2 = np.convolve(B, w2)


def q1():
    plt.plot(np.arange(A1.shape[0]), A1, color='r', label='A*WAVELET1')
    plt.plot(np.arange(A2.shape[0]), A2, color='k', label='A*WAVELET2')
    plt.plot(np.arange(B1.shape[0]), B1, color='b', label='B*WAVELET1')
    plt.plot(np.arange(B2.shape[0]), B2, color='g', label='B*WAVELET2')
    plt.legend()
    plt.show()

def q2():
    time_cross = np.correlate(A1, B1)
    print B1

    plt.plot(time_cross.shape[0], time_cross, color='k', label='cross correlation in time domain')
    plt.legend()
    plt.show()

q2()



