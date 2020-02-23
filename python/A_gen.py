#!/usr/bin/python3

from pylab import *
import numpy.matlib
import numpy as np
import numpy.linalg

def A_gen():
    shape = (3,3)

    yi, xi = np.indices(shape)
    i = np.ones(shape)
    A = np.dstack([i, xi, yi, xi*xi, xi*yi, yi*yi]).reshape((-1,6))
    #A /= 9*8
    print(A)

    At = np.transpose(A)

    #print(At)

    mul = np.matmul(At, A)
    #print(mul)

    i = inv(mul)
    #print(i)

    mul2  = np.matmul(i, At)

    #print(mul2)
    return mul2;


def mylstsq(b):
    mul2  = A_gen()

    return np.matmul(mul2, b)


if __name__ == "__main__":
    print(mylstsq([1,2,3,7,9,8,4,5,6]))
    res = A_gen()
    print(res);
    for r in res:
        for i in r:
            print(i, end='')
            print(", ", end='')
        print()