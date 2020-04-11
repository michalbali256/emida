#!/usr/bin/python3
from pylab import *

#N = 256


#s = 32
#positions = [ (i,j) for i in range(s,N-s+1,s) for j in range(s,N-s+1,s) ]

num_slices = int(input())



for line in sys.stdin:
    data = []
    name = line
    for i in range(num_slices):
        txt = input()
        x = txt.split()
        ints = [float(i) for i in x]
        data.append( (ints[0], ints[1], ints[2], ints[3]) )

    print(name)
    #print(data)

    data = array(data)


    N = 873
    center = N//2, N//2, N//2

    #
    # Recover transformation matrix from displacements
    # (It can only be determined up to a multiplicative factor)
    #

    a = data[:,0] - center[0]
    b = data[:,1] - center[1]
    z = center[2]

    c = a + data[:,2]
    d = b + data[:,3]

    #
    # Least squares (min |Ax-b|^2) with U[2,2] fixed to 1
    #
    A = zeros((2*len(data), 8))
    A[0::2,0] = -a*z
    A[0::2,1] = -b*z
    A[0::2,2] = -z*z

    A[1::2,3] = -a*z
    A[1::2,4] = -b*z
    A[1::2,5] = -z*z

    A[0::2,6] = a*c
    A[0::2,7] = b*c
    #A[0::2,8] = z*c

    A[1::2,6] = a*d
    A[1::2,7] = b*d
    #A[1::2,8] = z*d

    #print(A)

    B = zeros(2*len(data))
    B[0::2] = -z*c
    B[1::2] = -z*d

    v = np.linalg.lstsq(A, B, rcond=None)[0]
    UU = array(list(v)+[1,]).reshape((3,3))
    #print(UU)
    #print()

    #
    # Least squares (min |Ax|^2) wih |x|^2 = 1
    #
    A = zeros((2*len(data), 9))
    A[0::2,0] = -a*z
    A[0::2,1] = -b*z
    A[0::2,2] = -z*z

    A[1::2,3] = -a*z
    A[1::2,4] = -b*z
    A[1::2,5] = -z*z

    A[0::2,6] = a*c
    A[0::2,7] = b*c
    A[0::2,8] = z*c

    A[1::2,6] = a*d
    A[1::2,7] = b*d
    A[1::2,8] = z*d

    from scipy.linalg import svd
    v = svd(A)[2][8]
    UU = v.reshape((3,3))
    print(UU/UU[2,2])
    print()


