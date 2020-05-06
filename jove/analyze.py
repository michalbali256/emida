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

def numeric_fit(old, new, calib, F):
    pass

def post_process(data, calib, M):
    import time
    started = time.time()
    mask, Fs = [], []
    for d in data:
        print(".", end="", flush=True)
        xy, uv, q = d[:,:2], d[:,2:4], d[:,4:]

        # filter by determinant
        d = 4*q[:,3]*q[:,5]-q[:,4]*q[:,4]
        m = d > 1e-7

        if m.sum() > 10:
            F = algebraic_fit(xy[m], (xy+uv)[m], calib)
            mask.append(True)
            Fs.append(F)
        else:
            mask.append(False)


    F = np.asarray(Fs)
    mask = np.asarray(mask)

    # camera to sample coordinates
    #F = F.dot(M.T)

    # set volumetric scale to 1
    scale = np.cbrt(np.linalg.det(F))
    F /= scale[...,np.newaxis,np.newaxis]

    #
    # Finite strain theory
    #
    # Polar decomposition
    U, S, V = np.linalg.svd(F)
    R = U @ V
    #P = (V_T*S[:,np.newaxis,:]) @ V # == V.T @ diag(S) @ V

    V_T = V.transpose(0,2,1)
    R_T = R.transpose(0,2,1)
    epsilon = (V_T*np.log(S[:,np.newaxis,:])) @ V # == logm(P)

    k = (R - R_T)/2 # inverse or Rodrigez formula

    sintheta = np.sqrt((k*k).sum(axis=(1,2))/2)
    #costheta = (R.trace(0,1,2)-1)/2
    #theta = np.arctan2(sintheta, costheta)
    #omega = k*(theta/np.sin(theta))[:,np.newaxis,np.newaxis] # == logm(R)
    omega = k*(np.arcsin(sintheta)/sintheta)[:,np.newaxis,np.newaxis] # == logm(R)

    #
    # Infinitesimal strain theory
    #
    #F_T = F.transpose(0,2,1)
    #epsilon = (F+F_T)/2 - np.eye(3)[np.newaxis,:,:]
    #omega = (F-F_T)/2

    print("post_processed in", time.time()-started, "s")
    return mask, F, epsilon, omega

def apply_transform(F, old, calib):
    new = np.column_stack([old - calib[:2], np.repeat(calib[2], len(old))]).dot(F.T)
    return new[:,:2]/new[:,2,np.newaxis]*calib[2] + calib[:2]


class FQuiver:
    def __init__(self, xy, F, mask, calib, **kw):
        self.xy = xy
        self.F = F
        self.calib = calib
        self.kw = kw
        self.idx = np.full(len(mask), -1)
        self.idx[mask] = np.arange(len(F))

    def init(self, viewer):
        self.ax = viewer.ax2
        xy = self.xy[0]
        uv = self.get_uv(0)
        self.q = self.ax.quiver(xy[:,0], xy[:,1], uv[:,0], uv[:,1], animated=True, **self.kw)

    def get_uv(self, i):
        i = self.idx[i]
        if i == -1:
            return np.zeros_like(self.xy[i])
        return apply_transform(self.F[i], self.xy[i], self.calib) - self.xy[i]

    def update(self, i):
        uv = self.get_uv(i)
        self.q.set_UVC(uv[:,0], uv[:,1])
        self.ax.draw_artist(self.q)

class Ellipses:
    def __init__(self, xy, q, **kw):
        self.xy = xy
        self.q = q
        self.kw = kw

    def init(self, viewer):
        from matplotlib.collections import PolyCollection
        self.ax = viewer.ax2
        pos = self.get_ellipses(0)
        self.col = PolyCollection(pos, animated=True, **self.kw)
        self.ax.add_collection(self.col)

    def update(self, i):
        pos = self.get_ellipses(i)
        self.col.set_paths(pos)
        self.ax.draw_artist(self.col)

    def get_ellipses(self, i, n=12):
        from numpy import sqrt, pi, sin, cos, arctan, where, linspace, stack, newaxis
        q = self.q[i]
        xy = self.xy[i]

        G = q[:,0]
        D, F = q[:,1]/2, q[:,2]/2
        A, B, C = q[:,3], q[:,4]/2, q[:,5]

        a = sqrt(2*(A*F*F+C*D*D+G*B*B-2*B*D*F-A*C*G)/(B*B-A*C)/(+sqrt((A-C)*(A-C)+4*B*B)-(A+C)))
        b = sqrt(2*(A*F*F+C*D*D+G*B*B-2*B*D*F-A*C*G)/(B*B-A*C)/(-sqrt((A-C)*(A-C)+4*B*B)-(A+C)))
        t = arctan(2*B/(A-C))/2 + where(A>C, pi/2, 0)

        tt = linspace(0, 2*pi, n, endpoint=False)
        xx = a[:,newaxis]*sin(tt[newaxis,:])
        yy = b[:,newaxis]*cos(tt[newaxis,:])

        s, c = sin(t[:,newaxis]), cos(t[:,newaxis])
        w = stack([xx* c + yy*-s,
                   xx*s + yy*c], axis=2)

        data = xy[:,newaxis,:] + w
        return data

class Cor:
    def __init__(self, dset):
        self.dset = dset

    def init(self, viewer):
        self.ax = viewer.ax2
        s = self.dset.roi.size
        fname = self.dset.fnames[0]
        self.dset.get_ref()
        limits = self.ax.axis()
        self.imgs = [
                self.ax.imshow(cor, extent=(j-2*s-.5,j+2*s-.5,i+2*s-.5,i-2*s-.5), vmin=-1e16, vmax=1e16)
                for (j,i), (cor, xp, yp, q) in zip(self.dset.roi.positions, self.dset.get_cor(fname, fit_size=3)) ]
        self.ax.axis(limits)

    def update(self, i):
        fname = self.dset.fnames[i]
        for img, (cor, xp, yp, q) in zip(self.imgs, self.dset.get_cor(fname, fit_size=3)):
            img.set_data(cor)
            self.ax.draw_artist(img)


if __name__ == "__main__":
    from run import *

    from run import deformed

    #data = load_result("out-initial-jove-5.txt")

    dset = deformed#.decimate(5)
    data = dset.load_result("out-deformed-jove.txt")

    bg = dset.load_ang()

    calib = np.array([48.8235, 77.5223, 69.8357])
    calib = 873*(np.array([0,1,0]) + np.array([1,-1,1])*calib/100)
    M = np.eye(3)
    mask, F, epsilon, omega = post_process(data, calib, M)


    from viewer import *
    v2 = Viewer(pos=dset.pos, actors=[
        HexBg(bg[:,:2], bg[:,2], cmap='gray'),
        Dots(dset.pos),
        Cursor(dset.pos),
        Img(dset.fnames),

        Quiver(data[:,:,:2], data[:,:,2:4], color="r", angles='xy', scale_units='xy', scale=0.1),
        FQuiver(data[:,:,:2], F, mask, calib, color="b", angles='xy', scale_units='xy', scale=0.1),
        Ellipses(data[:,:,:2] + 10*data[:,:,2:4], data[:,:,4:],  facecolor='none', edgecolor='g'),

        #Cor(dset),
        #Ellipses(data[:,:,:2]+data[:,:,2:4], data[:,:,4:],  facecolor='none', edgecolor='g'),
    ])

    from matplotlib.pyplot import figure, savefig
    dpi = int(np.sqrt(len(dset.pos))*4)

    vlim = min(200e-4, np.nanmax(np.abs(epsilon)))

    fig = figure(figsize=(6,6))
    for i,j in [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]:
        ax = fig.add_axes((j/3, (2-i)/3, 1/3, 1/3))
        dset.plot(dset.pos[mask], epsilon[:,i,j], vmin=-vlim, vmax=vlim, cmap='jet', ax=ax)
        ax.axis("off")
    savefig("epsilon.png", dpi=dpi)

    vlim = min(200e-4, np.nanmax(np.abs(omega)))

    fig = figure(figsize=(6,6))
    for i,j in [(0,1), (0,2), (1,2)]:
        ax = fig.add_axes((j/3, (2-i)/3, 1/3, 1/3))
        dset.plot(dset.pos[mask], omega[:,i,j], vmin=-vlim, vmax=vlim, cmap='jet', ax=ax)
        ax.axis("off")
    savefig("omega.png", dpi=dpi)
    show()
