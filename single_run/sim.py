#!/usr/bin/python3

from pylab import *

def vlen(v):
    return sqrt(einsum("...i,...i->...", v, v))

def circle(v, b=0, npoints=500):
    r2, r3 = vlen(v[:2]), vlen(v)
    
    cp, sp = v[0]/r2, v[1]/r2
    st, ct = r2/r3, v[2]/r3

    def rot(c,s,x,y):
        return c*x-s*y, s*x+c*y

    a = linspace(-pi, pi, npoints)
    #a = linspace(0, pi, npoints)
    x = sin(a)*cos(b) 
    y = cos(a)*cos(b)
    z = sin(b)

    x,z = rot(ct, st, x, z)    
    x,y = rot(-cp, -sp, x, y)

    return transpose([x,y,z])

def euler2rot(angles):
    phi1, PHI, phi2 = asarray(angles)/180*pi
    R = zeros((3,3))
    R[0,0] = cos(phi1)*cos(phi2) - sin(phi1)*sin(phi2)*cos(PHI)
    R[0,1] = sin(phi1)*cos(phi2) + cos(phi1)*sin(phi2)*cos(PHI)
    R[0,2] = sin(phi2)*sin(PHI)

    R[1,0] = -cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(PHI) 
    R[1,1] = -sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(PHI)
    R[1,2] = cos(phi2)*sin(PHI)

    R[2,0] = sin(phi1)*sin(PHI)
    R[2,1] = -cos(phi1)*sin(PHI)
    R[2,2] = cos(PHI)
    return R

def sim_pattern(R, calib, M=4, w=0.01):
    v = mgrid[-M:M+1,-M:M+1,-M:M+1].T.reshape((-1, 3))
    v = v[ vlen(v) <= M ]
    v = v[ v.sum(axis=-1)%2 == 0 ]

    cx,cy,cz = calib
    plot([cx],[cy],"r+") 
    for vv in v.dot(R):
        vl = vlen(vv)
        if vl == 0:
            continue
        p = circle(vv, w*vl)
        m = p[:,2] > 0
        p[~m] = np.nan
        plot(cx+p[:,0]/p[:,2]*cz, cy+p[:,1]/p[:,2]*cz, 'r-', lw=1/vl)

def poles(R, calib, M=4):
    v = mgrid[-M:M+1,-M:M+1,-M:M+1].T.reshape((-1, 3))
    v = v[ vlen(v) <= M ]
    cx, cy, cz = calib
    v = v.dot(R)
    v = v[v[:,2]>0]
    return cx+v[:,0]/v[:,2]*cz, cy+v[:,1]/v[:,2]*cz
 


a=23/180*pi;
# x and y cols swapped
M = array([[0, cos(a), sin(a)],
           [1, 0,      0],
           [0,-sin(a), cos(a)]])



if __name__ == "__main__":
    data = array([
        (-2,0,-1,  17,  61),
        (-1,0,-1,  255, 19),
        (-2,1,-1,  125, 347),
        (-1,1,-1,  350, 403),
        (-1,1,-2,  498, 245),
        (0,1,-2,   791, 266),
        (-1,2,-1,  414, 662),
        (0,1,-1,   784, 512),
        (0,2,-1,   775, 839),
    ])
    hkl = data[:,:3]
    xy = data[:,3:]

    def func( p):
        R = euler2rot(p[:3]).dot(M)
        v = hkl.dot(R)
        return concatenate([
            v[:,0]/v[:,2]*p[5]+p[3] - xy[:,0], 
            v[:,1]/v[:,2]*p[5]+p[4] - xy[:,1],
        ])

    from scipy.optimize import leastsq
    calib = array([48.8235, 77.5223, 69.8357])
    angles = array([2.1, 150.8, 261.5])

    calib = 873*(array([0,1,0]) + array([1,-1,1])*calib/100)

    p0 = concatenate([angles, calib])
    p = p0
    p = leastsq(func, p0)[0]
    print(p)

    R = euler2rot(p[:3]).dot(M)
    v = hkl.dot(R)
    x = v[:,0]/v[:,2]*p[5]+p[3]
    y = v[:,1]/v[:,2]*p[5]+p[4]
 
    print("angles", angles)
    print("angles", p[:3])

    #A = euler2rot(angles)
    #B = euler2rot(p[:3])
    #print("R", A)
    #print("R", B)

    print("calib", calib)
    print("calib", p[3:])

    from PIL import Image
    figure(figsize=(10,10))
    tight_layout()
    #fname = "saved-autotune-g.png"
    fname = "../../Testing data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif"
    imshow( Image.open(fname), cmap="gray")
    axis("image")
    grid(False)

    plot(xy[:,0], xy[:,1], "b+")
    plot(x, y, "rx")
    sim_pattern( euler2rot(p[:3]).dot(M), calib, w=0.015)
    savefig("plot.pdf")
    show()
