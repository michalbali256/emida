#!/usr/bin/python3

from pylab import *
import numpy.matlib
import numpy as np
import numpy.linalg

common_denom = 198*25*8*9*35
def A_gen():
    shape = (9,9)

    yi, xi = np.indices(shape)
    i = np.ones(shape)
    A = np.dstack([i, xi, yi, xi*xi, xi*yi, yi*yi]).reshape((-1,6))
    print(A)
    A /= common_denom
    print(A)

    At = np.transpose(A)

    #print(At)

    mul = np.matmul(At, A)
    #print(mul)

    i = inv(mul)
    #print(i)

    mul2  = np.matmul(i, At)
    
    rounded = np.round(mul2, 0)
    print (mul2)
    
    delta = mul2 - rounded
    for r in delta:
        for i in r:
            if i > 1E-9:
                print ('Error ', i)
    #print(mul2)
    return rounded


def mylstsq(b):
    mul2  = A_gen()

    return np.matmul(mul2, b)


if __name__ == "__main__":
    #print(mylstsq([1,2,3,7,9,8,4,5,6]))
    res = A_gen()
    print(res)
    for r in res:
        for i in r:
            print(i, end='')
            print('/', end='')
            print(common_denom, end='')
            print(", ", end='')
        print()