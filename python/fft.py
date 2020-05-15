from pylab import *
import numpy.matlib
import numpy as np
import numpy.linalg
import numpy.fft
from scipy.signal import correlate
import cmath

np.set_printoptions(suppress=True,linewidth=np.nan)

sh = [cmath.rect(1,2*cmath.pi/8*i*5) for i in range(0,5)]
sh = [math.cos(2*cmath.pi/8*i*5) + math.sin(2*cmath.pi/8*i*5)*1j for i in range(0,5)]
sy =np.array([cmath.rect(1,2*cmath.pi/8*i*5) for i in range(0,8)])

a = [[1,2,3,1],[4,5,6,1],[7,8,9,1],[7,8,9,1]]
b = [[6,3,2,1],[12,23,3,1],[1,8,9,1],[7,8,9,1]]

a_and_zeros = [[1,2,3,1,0,0,0,0],[4,5,6,1,0,0,0,0],[7,8,9,1,0,0,0,0],[7,8,9,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]]
b_and_zeros = [[6,3,2,1,0,0,0,0],[12,23,3,1,0,0,0,0],[1,8,9,1,0,0,0,0],[7,8,9,1,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]]

#print(a_and_zeros)
#a_and_zeros = a.copy()
#a_and_zeros.extend([0] * len(a))

#b_and_zeros = b.copy()
#b_and_zeros.extend([0] * len(b))
#b_and_zeros.reverse()

ffta = np.fft.rfft2(a_and_zeros)
fftb = np.fft.rfft2(b_and_zeros)
print(ffta)
print()

#print(fftb)
#print()
mult = ffta * np.conj(fftb)
mult *= sh
mult *= sy[:, np.newaxis]
ia = np.fft.irfft2(mult)
print(ia)

cor = correlate(a, b, mode='full')
print(cor)