#!/usr/bin/python3
from pylab import *

N = 256
center = N//2, N//2, N//2

a = 0.001
#U = array([[1,0,0],[0,1,0],[0,0,1]])

#U = array([[1,a,0],[0,1,0],[0,0,1]])
#U = array([[1,0,0],[a,1,0],[0,0,1]])
#U = array([[1,0,a],[0,1,0],[0,0,1]])
#U = array([[1,0,0],[0,1,a],[0,0,1]])
#U = array([[1,0,0],[0,1,0],[a,0,1]])
#U = array([[1,0,0],[0,1,0],[0,a,1]])

#U = array([[1,a,0],[a,1,0],[0,0,1]])
U = array([[1,-a,0],[a,1,0],[0,0,1]])
#U = array([[cos(a),sin(a),0],[-sin(a),cos(a),0],[0,0,1]])

#U = array([[1+a,0,0],[0,1,0],[0,0,1]])
#U = array([[1,0,0],[0,1+a,0],[0,0,1]])
#U = array([[1,0,0],[0,1,0],[0,0,1+a]])
#U = array([[1+a,0,0],[0,1+a,0],[0,0,1+a]])

#U = eye(3) + normal(size=(3,3))/10000

def func(x, y, center, p): # perspective projection
    x = x - center[0]
    y = y - center[1]
    z = center[2]
    xx = x*p[0,0] + y*p[0,1] + z*p[0,2]
    yy = x*p[1,0] + y*p[1,1] + z*p[1,2]
    zz = x*p[2,0] + y*p[2,1] + z*p[2,2]
    return xx/zz*center[2] + center[0], yy/zz*center[2] + center[1]

def generate():

    #
    # This is to be determined
    #

    print(U/U[2,2])
    print()

    #
    # Make reference and transformed image (your inputs)
    #
    reftype = 'sim'

    if reftype == 'chess':
        # chessbard
        ref = fromfunction(lambda i,j: (i//8+j//8)%2, (N,N)).astype(float)

        from scipy.ndimage import map_coordinates
        y, x = indices(ref.shape)
        new_x, new_y = func(x, y, center, U)
        tran = map_coordinates(ref, [new_y, new_x], order=1)

    elif reftype == 'blobs':
        # random blobs
        from scipy.ndimage import gaussian_filter
        ref = (gaussian_filter(normal(size=(N,N)), 1) > 0).astype(float)

        from scipy.ndimage import map_coordinates
        y, x = indices(ref.shape)
        new_x, new_y = func(x, y, center, U)
        tran = map_coordinates(ref, [new_y, new_x], order=1)

    elif reftype == 'sim':
        # random rotation matrix
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        w,x,y,z = normal(size=4)
        n = w*w + x*x + y*y + z*z
        s = 0 if n == 0 else 2/n
        R = array([[1-s*y*y-s*z*z,   s*x*y-s*w*z,   s*x*z+s*w*y],
                [  s*x*y+s*w*z, 1-s*x*x-s*z*z,   s*y*z-s*w*x],
                [  s*x*z-s*w*y,   s*y*z+s*w*x, 1-s*x*x-s*y*y]])
    
        R = array([ # fixed for testing
        [ 0.53755579, -0.22236491, -0.81338036],
        [-0.74554228,  0.32532616, -0.58166107],
        [ 0.39395492,  0.91908473,  0.00909852]])
    
        # simulated pattern
        def sim_pattern(N, center, R=None, M=4, w=30, s=3.5, step=32):
            v = mgrid[-M:M+1,-M:M+1,-M:M+1].T.reshape((-1, 3))
            u = indices((N,N,1)).T.reshape((N,N,3)) - center

            if R is not None:
                v = v.dot(R)

            ulen = sqrt((u*u).sum(axis=-1))[:,:,newaxis]
            vlen = sqrt((v*v).sum(axis=-1))[newaxis,newaxis,:]
    
            ret = empty((N,N))
            for i in range(0, N, step):
                print(".", end="", flush=True)
                d = (u[i:i+step,:,newaxis,:]*v[newaxis,newaxis,:,:]).sum(axis=-1)/ulen[i:i+step]/vlen
                ret[i:i+step] = ( (abs(d)<vlen/w)*exp(-vlen*vlen/s/s) ).sum(axis=-1)
                del d
            print()
            return ret

        if 0: 
            img = sim_pattern(2048, (1024, 1024, 1024), R)
            #img = sim_pattern(N, center, R)
            from PIL import Image
            Image.fromarray((img/img.max()*65535).astype("uint16")).save("ref.tif")

        ref = sim_pattern(N, center, R)
        tran = sim_pattern(N, center, R.dot(U))

        if 0: # blur
            from scipy.ndimage import gaussian_filter
            ref = gaussian_filter(ref, 1)
            tran = gaussian_filter(tran, 1)

        if 0: # noise
            ref = poisson(ref*1000)
            tran = poisson(tran*1000)
    
    return ref, tran
