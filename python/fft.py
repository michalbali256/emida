from pylab import *
import numpy.matlib
import numpy as np
import numpy.linalg
import numpy.fft
from scipy.signal import correlate

np.set_printoptions(suppress=True,linewidth=np.nan)

a = [[1,2,3,1],[4,5,6,1],[7,8,9,1],[7,8,9,1]]
b = [[6,3,2,1],[12,23,3,1],[1,8,9,1],[7,8,9,1]]

a_and_zeros = [[1,2,3,1,0,0,0,0],[4,5,6,1,0,0,0,0],[7,8,9,1,0,0,0,0],[7,8,9,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]]
b_and_zeros = [[6,3,2,1,0,0,0,0],[12,23,3,1,0,0,0,0],[1,8,9,1,0,0,0,0],[7,8,9,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]]

print(a_and_zeros)
#a_and_zeros = a.copy()
#a_and_zeros.extend([0] * len(a))

#b_and_zeros = b.copy()
#b_and_zeros.extend([0] * len(b))
#b_and_zeros.reverse()

ffta = np.fft.rfft2(a_and_zeros)
fftb = np.fft.rfft2(b_and_zeros)
print(ffta)
print()
print(fftb)
print()
print(ffta * np.conj(fftb))
ia = np.fft.irfft2(ffta * np.conj(fftb))
print(ia)

cor = correlate(a, b, mode='full')
print(cor)