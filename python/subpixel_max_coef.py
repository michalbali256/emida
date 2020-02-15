from pylab import *
import numpy.matlib
import numpy as np
import numpy.linalg

l=np.asarray([[1,2,3],[4,5,6],[7,8,9]])
shape = (3,3);
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
b = l.reshape(-1)
print(b)
#f(x,y) = a + bx + cy + dx^2 + exy + fy^2
q = np.linalg.lstsq(A, b, rcond=None)[0]

print(q)