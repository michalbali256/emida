#!/usr/bin/python3

import numpy as np
from scipy.linalg import logm

def algebraic_fit(old, new, calib):
    assert len(old) == len(new)
    assert len(old) > 4

    oldx, oldy = (old - calib[:2]).T
    newx, newy = (new - calib[:2]).T
    z = calib[2]

    A = np.zeros((2*len(old), 9))
    A[0::2, 0] = -oldx*z
    A[0::2, 1] = -oldy*z
    A[0::2, 2] = -z*z

    A[1::2, 3] = -oldx*z
    A[1::2, 4] = -oldy*z
    A[1::2, 5] = -z*z

    A[0::2, 6] = oldx*newx
    A[0::2, 7] = oldy*newx
    A[0::2, 8] = z*newx

    A[1::2, 6] = oldx*newy
    A[1::2, 7] = oldx*newy
    A[1::2, 8] = z*newy

    U, S, V = np.linalg.svd(A)
    return V[-1].reshape(3,3)


def process(data, calib, M):
    import time
    started = time.time()
    Fs, epsilon, omega = [], [], []
    for d in data:
        print(".", end="", flush=True)
        xy, uv, q = d[:,:2], d[:,2:4], d[:,4:]

        # filter by determinant
        d = 4*q[:,3]*q[:,5]-q[:,4]*q[:,4]
        m = d > 1e-7

        if m.sum() > 4:
            F = algebraic_fit(xy[m], (xy+uv)[m], calib)
        else:
            F = np.eye(3)

        # set volumetric scale to 1
        d = np.linalg.det(F)
        F /= np.sign(d)*abs(d)**(1/3)

        # get from camera to sample coordinates
        F = M.dot(F)

        # polar decomposition
        U,S,V = np.linalg.svd(F)
        R = U.dot(V)
        P = V.T.dot(S[:,np.newaxis]*V) # == diag(S).dot(V)

        Fs.append( F )
        epsilon.append( logm(P) )
        omega.append( logm(R) )
    print("processed in", time.time()-started, "s")
    return np.asarray(Fs), np.asarray(epsilon), np.asarray(omega)


def apply_transform(F, old, calib):
    new = np.column_stack([old - calib[:2], np.repeat(calib[2], len(old))]).dot(F.T)
    return new[:,:2]/new[:,2,np.newaxis]*calib[2] + calib[:2]

class FQuiver:
    def __init__(self, xy, F, calib, **kw):
        self.xy = xy
        self.F = F
        self.calib = calib
        self.kw = kw

    def init(self, viewer):
        self.ax = viewer.ax2
        xy = self.xy[0]
        uv = self.get_uv(0)
        self.q = self.ax.quiver(xy[:,0], xy[:,1], uv[:,0], uv[:,1], animated=True, **self.kw)

    def get_uv(self, i):
        return apply_transform(self.F[i], self.xy[i], self.calib) - self.xy[i]

    def update(self, i):
        uv = self.get_uv(i)
        self.q.set_UVC(uv[:,0], uv[:,1])
        self.ax.draw_artist(self.q)

if __name__ == "__main__":
    #from run import initial as dset
    from run import deformed as dset

    from numpy import loadtxt
    bg = loadtxt(dset.ang, usecols=(3,4,5))

    from tools import load_result
    #pos, fnames, data = load_result("out-initial-jove-5.txt")
    #pos, fnames, data = load_result("out-initial-jove.txt")
    pos, fnames, data = load_result("out-deformed-jove-5.txt")
    #pos, fnames, data = load_result("out-deformed-jove.txt")

    calib = np.array([48.8235, 77.5223, 69.8357])
    calib = 873*(np.array([0,1,0]) + np.array([1,-1,1])*calib/100)
    M = np.eye(3)

    F, epsilon, omega = process(data, calib, M)

    from viewer import *
    v2 = Viewer(pos=pos, actors=[
        HexBg(bg[:,:2], bg[:,2], cmap='gray'),
        Dots(pos),
        Cursor(pos),
        Img(fnames),
        Quiver(data[:,:,:2], data[:,:,2:4], color="r", angles='xy', scale_units='xy', scale=0.1),
        FQuiver(data[:,:,:2], F, calib, color="b", angles='xy', scale_units='xy', scale=0.1)
    ])


    from matplotlib.pyplot import figure, savefig

    vlim = min(300e-4, abs(epsilon).max())

    fig = figure(figsize=(6,6))
    for i,j in [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]:
        ax = fig.add_axes((j/3, (2-i)/3, 1/3, 1/3))
        dset.plot(pos, epsilon[:,i,j], vmin=-vlim, vmax=vlim, cmap='jet', ax=ax)
        ax.axis("off")
    savefig("epsilon.png", dpi=300)

    vlim = min(300e-4, abs(omega).max())

    fig = figure(figsize=(6,6))
    for i,j in [(0,1), (0,2), (1,2)]:
        ax = fig.add_axes((j/3, (2-i)/3, 1/3, 1/3))
        dset.plot(pos, omega[:,i,j], vmin=-vlim, vmax=vlim, cmap='jet', ax=ax)
        ax.axis("off")
    savefig("omega.png", dpi=300)

    show()
