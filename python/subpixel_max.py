from pylab import *
import numpy.matlib
import numpy as np
import numpy.linalg



def subpixel_peak(d, s=1):
    """Fit polynomial quadratic in x and y to the neighbourhood of maximum,
       to determine maximum with subpixel precision"""
    #find indices with the maximum
    yp, xp = np.unravel_index(np.argmax(d), d.shape)
    y, x = yp-s, xp-s
    l = d[y:y+2*s+1, x:x+2*s+1]
    print(l)
    #setup matrix A for least squares
    yi, xi = np.indices(l.shape)
    #print("l example") 
    #print(l)
    i = np.ones(l.shape)
    A = np.dstack([i, xi, yi, xi*xi, xi*yi, yi*yi]).reshape((-1,6))
# A:
#[[1. 0. 0. 0. 0. 0.]
# [1. 1. 0. 1. 0. 0.]
# [1. 2. 0. 4. 0. 0.]
# [1. 0. 1. 0. 0. 1.]
# [1. 1. 1. 1. 1. 1.]
# [1. 2. 1. 4. 2. 1.]
# [1. 0. 2. 0. 0. 4.]
# [1. 1. 2. 1. 2. 4.]
# [1. 2. 2. 4. 4. 4.]]
    #print("A expample") 
    #print(A)
    b = l.reshape(-1)
    
    #f(x,y) = a + bx + cy + dx^2 + exy + fy^2
    q = np.linalg.lstsq(A, b, rcond=None)[0]
    
    print(q)

    #from partial derivation by x: 2dx + ey + b = 0
    #from partial derivation by y: ex + 2fy + c = 0
    xs, ys = np.linalg.solve([[2*q[3],   q[4]],
                              [  q[4], 2*q[5]]],
                            [-q[1], -q[2]])
    return y+ys, x+xs 

d=np.asarray([[1,2,3],[7,9,8],[4,5,6]])

y, x = subpixel_peak(d, 1)

print(x)
print(y)