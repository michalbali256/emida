#!/usr/bin/python3

from pylab import *
from PIL import Image

calib = array([48.8235, 77.5223, 69.8357])
calib = 873*(array([0,1,0]) + array([1,-1,1])*calib/100)
from tools import hexplot

from scipy.linalg import svd, logm
def fit_transform(old, new, calib):
    cx,cy,cz = calib
    oldx, oldy = (old - (cx,cy)).T
    newx, newy = (new - (cx,cy)).T

    A = zeros((2*old.shape[0], 9))
    A[0::2, 0] = -oldx*cz
    A[0::2, 1] = -oldy*cz
    A[0::2, 2] = -cz*cz

    A[1::2, 3] = -oldx*cz
    A[1::2, 4] = -oldy*cz
    A[1::2, 5] = -cz*cz

    A[0::2, 6] = oldx*newx
    A[0::2, 7] = oldy*newx
    A[0::2, 8] = cz*newx

    A[1::2, 6] = oldx*newy
    A[1::2, 7] = oldx*newy
    A[1::2, 8] = cz*newy

    u,s,v = svd(A)
    F = v[-1].reshape(3,3)
    d = linalg.det(F)
    F /= sign(d)*abs(d)**(1/3)

    U,S,V = svd(F)
    R = U.dot(V)
    P = V.T.dot(diag(S).dot(V))

    return F, logm(R), logm(P), s[-1]

def apply_transform(T, old, calib):
    cx,cy,cz = calib
    new = column_stack([old - (cx,cy), full(old.shape[0], cz)]).dot(T.T)
    return new[:,:2]/new[:,2,newaxis]*cz + (cx,cy)

xs, ys, vs, fnames, data1s, data2s, Fs, Rs, Ps = [], [], [], [], [], [], [], [], []
mask = asarray(Image.open("mask.png"))==255
with open("out-jove-300.txt") as f1, open("out-emida-300.txt") as f2:
#with open("out-jove-60.txt") as f1, open("out-emida-60.txt") as f2:
    while True:
        print(".",end="",flush=True)
        d1 = f1.readline().rstrip().split(maxsplit=3)
        d2 = f2.readline().rstrip().split(maxsplit=3)
        if not d1:
            break
        assert d1 == d2
        x, y, n, fname = d1
        x, y, n = float(x), float(y), int(n)
        data1 = loadtxt(f1, max_rows=n)
        data2 = loadtxt(f2, max_rows=n)
        #fname = fname[6:]

        q = data1[:,4:]
        dt = 4*q[:,3]*q[:,5]-q[:,4]*q[:,4]
        m = dt>1e15
        try:
            F, R, P, v = fit_transform(data1[m,:2], data1[m,:2]+data1[m,2:4], calib)
        except:
            import traceback
            traceback.print_exc()
            continue
        v = log(v)
        data1[~m,2:4] = nan
        data2[~m,2:4] = nan
        #a = array(Image.open(fname))*mask
        #v = a[:a.shape[0]//2].sum()-a[a.shape[0]//2:].sum()

        xs.append(x)
        ys.append(y)
        data1s.append(data1)
        data2s.append(data2)
        fnames.append(fname)
        Fs.append(F)
        Rs.append(R)
        Ps.append(P)
        vs.append(v)
print()


xs = array(xs)
ys = array(ys)
vs = array(vs)
Fs = array(Fs)
Rs = array(Rs)
Ps = array(Ps)


fig, sub = subplots(3,6, gridspec_kw=dict(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1), figsize=(12,6))
for i,j in ndindex(3,6):
    ax = sub[i,j]
    if j < 3:
        if j<i:
            ax.remove()
            continue
        v = Ps[:,i,j]
    else:
        if j-3<=i:
            ax.remove()
            continue
        v = Rs[:,i,j-3]
    hexplot(column_stack([xs,ys]), v, vmin=-200e-4, vmax=200e-4, cmap='jet', ax=ax)
#    ax.tripcolor(xs, ys, v, shading='gouraud', vmin=-200e-4, vmax=200e-4, cmap='jet')
#    ax.plot(xs, ys, "k,")
    #ax.set_xticklabels('')
    #ax.set_yticklabels('')
    ax.axis("off")
    ax.set_aspect("equal")
#fig.tight_layout()


fig = figure(figsize=(10,5))
ax1, ax2 = fig.subplots(1,2)
col = hexplot(column_stack([xs,ys]), vs, ax=ax1)
l1, = ax1.plot([],[],"rx", ms=10, animated=True)
ax1.set_aspect("equal")
ax1.grid(False)
ax2.grid(False)

fname = fnames[0]
im2 = ax2.imshow(Image.open(fname)*mask)

data = data1s[0]
qv1 = ax2.quiver(data[:,0], data[:,1], data[:,2], data[:,3], color="g", angles='xy', alpha=0.7, scale_units='xy', scale=0.1)

data = data2s[0]
qv2 = ax2.quiver(data[:,0], data[:,1], data[:,2], data[:,3], color="r", angles='xy', alpha=0.7, scale_units='xy', scale=0.1)

new = apply_transform(Fs[0],  data[:,:2], calib)
qv3 = ax2.quiver(data[:,0], data[:,1], new[:,0]-data[:,0], new[:,1]-data[:,1], color="b", angles='xy', alpha=0.7, scale_units='xy', scale=0.1)

fig.tight_layout()

#show(block=False)
#fig.canvas.draw()

def draw_handler(ev):
    fig._ax1back = fig.canvas.copy_from_bbox(ax1.bbox)

last = 0
def motion_handler(ev):
    if ev.inaxes==ax1:
        i = argmin( hypot(xs-ev.xdata, ys-ev.ydata))
        global last
        if i == last:
            return
        last = i
        l1.set_data([xs[i]],[ys[i]])
        fname = fnames[i]
        print(fname)
        print(Rs[i])
        print(Ps[i], trace(Ps[i]))
        im2.set_data(Image.open(fname)*mask)

        data = data1s[i]
        qv1.set_UVC(data[:,2], data[:,3])

        data = data2s[i]
        qv2.set_UVC(data[:,2], data[:,3])

        new = apply_transform(Fs[i],  data[:,:2], calib)
        qv3.set_UVC(new[:,0]-data[:,0], new[:,1]-data[:,1])


        fig.canvas.restore_region(fig._ax1back)
        ax1.draw_artist(l1)
        ax2.draw_artist(im2)
        ax2.draw_artist(qv1)
        ax2.draw_artist(qv2)
        ax2.draw_artist(qv3)

        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)

fig.canvas.mpl_connect('motion_notify_event', motion_handler)
fig.canvas.mpl_connect('draw_event', draw_handler)
show()
