#!/usr/bin/python3
from pylab import *

from generator import *

#N = 256
#center = N//2, N//2, N//2

ref, tran = generate()
figure(figsize=(8,8))
subplot(221)
imshow(ref, cmap='gray')

subplot(222)
imshow(tran, cmap='gray')

subplot(223)
a,b = ref.min(), ref.max()
c,d = tran.min(), tran.max()
rgb = dstack([(ref-a)/(b-a), (tran-c)/(d-c), zeros(ref.shape)])
imshow(rgb)

from A_gen import *

#
# Find local displacements from crosscorrelations, this is to be implemented on GPU
#
from scipy.signal import correlate
def subpixel_peak(d, s=1):
    """Fit polynomial quadratic in x and y to the neighbourhood of maximum,
       to determine maximum with subpixel precision"""
    #find indices with the maximum
    #print(d)
    #print(np.argmax(d))
    yp, xp = np.unravel_index(np.argmax(d), d.shape)
    #print(xp, yp)
    y, x = yp-s, xp-s
    l = d[y:y+2*s+1, x:x+2*s+1]
    
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

    b = l.reshape(-1)
    
    #f(x,y) = a + bx + cy + dx^2 + exy + fy^2
    q = np.linalg.lstsq(A, b, rcond=None)[0]
    #q1 = mylstsq(b)

    #from partial derivation by x: 2dx + ey + b = 0
    #from partial derivation by y: ex + 2fy + c = 0
    xs, ys = np.linalg.solve([[2*q[3],   q[4]],
                              [  q[4], 2*q[5]]],
                            [-q[1], -q[2]])
    return y+ys, x+xs 

#s = 16
#positions = [ (i,j) for i in range(s,N-s+1,2*s) for j in range(s,N-s+1,2*s) ]

s = 32
positions = [ (i,j) for i in range(s,N-s+1,s) for j in range(s,N-s+1,s) ]

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
ax = gca()
areas = [ Rectangle( (j-s, i-s), 2*s, 2*s) for i,j in positions ]
gca().add_collection(PatchCollection(areas, facecolor='none', edgecolor='b'))
    
data = []
window = hanning(2*s)
#print('POSITIONS')
for i,j in positions:
    a = ref[i-s:i+s, j-s:j+s]
    b = tran[i-s:i+s, j-s:j+s] 

    a = a - a.mean()
    b = b - b.mean()
    #print(a)
    #print(b)
    a *= window[:,newaxis]
    a *= window[newaxis,:]
    b *= window[:,newaxis]
    b *= window[newaxis,:]
    cor = correlate(a, b, mode='full')

    yp, xp = subpixel_peak(cor)
    #print(xp, yp)
    print('{0:.30f}'.format(xp-2*s+1), '{0:.30f}'.format(yp-2*s+1))
    if 0:
        figure()
        subplot(221)
        imshow(a)
        subplot(222)
        imshow(b)
        subplot(223)
        imshow(cor)
        show()

    data.append( (j, i, xp-2*s+1, yp-2*s+1) )

data = array(data)

#
# Compare real and estimated displacements
#
subplot(224)
new_x, new_y = func(data[:,0], data[:, 1], center, U)
quiver(data[:,0], data[:,1], new_x-data[:,0], new_y-data[:,1], color="r", angles='xy', alpha=0.7)
quiver(data[:,0], data[:,1], data[:,2], data[:,3], color="b", angles='xy', alpha=0.7)
gca().set_aspect("equal")
ylim(N,0)


#
# Recover transformation matrix from displacements
# (It can only be determined up to a multiplicative factor)
#

if 0: # test with ideal data 
    data[:,2] = new_x - data[:,0] + normal(0, 0.01, size=new_x.shape)
    data[:,3] = new_y - data[:,1] + normal(0, 0.01, size=new_y.shape)

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

new_x, new_y = func(data[:,0], data[:, 1], center, UU)
quiver(data[:,0], data[:,1], new_x-data[:,0], new_y-data[:,1], color="g", angles='xy', alpha=0.7)

"""
def opt(p):
    p = p.reshape((3,3))
    #print(p, end=" ")
    
    new_x, new_y = func(data[:,0], data[:,1], center, p)
    ret = ravel([ new_x-data[:,0] - data[:,2], 
                  new_y-data[:,1] - data[:,3] ])
    #print((ret**2).sum())
    print(".", end="", flush=True)
    return ret

from scipy.optimize import leastsq
UUU, r = leastsq(opt, UU) # nonlinear fit to correctly measure distances
print()
UUU = UUU.reshape((3,3))
print(UUU/UUU[2,2])
print()
"""

tight_layout()
show()
