#!/usr/bin/python3

from pylab import *
from PIL import Image

xs, ys, vs, fnames, data1s, data2s = [], [], [], [], [], []
mask = asarray(Image.open("mask.png"))==255
with open("out-jove.txt") as f1, open("out-emida.txt") as f2:
    while True:
        print(".",end="",flush=True)
        d1 = f1.readline().rstrip().split(maxsplit=3)
        d2 = f2.readline().rstrip().split(maxsplit=3)
        if not d1:
            break
        assert d1 == d2
        x, y, n, fname = d1
        x, y, n = float(x), float(y), int(n)
        data1 = loadtxt(f1,max_rows=n)
        data2 = loadtxt(f2,max_rows=n)
        #fname = fname[6:]

        xs.append(x)
        ys.append(y)
        data1s.append(data1)
        data2s.append(data2)
        fnames.append(fname)

        a = array(Image.open(fname))*mask
        vs.append( a[:a.shape[0]//2].sum()-a[a.shape[0]//2:].sum())
print()


xs = array(xs)
ys = array(ys)
vs = array(vs)
from pylab import figure, show

fig = figure(figsize=(10,5))
ax1, ax2 = fig.subplots(1,2)
ax1.scatter(xs, ys, c=vs)
#l1, = ax1.plot([xs[0]],[xs[1]],"rx",ms=10)
ax1.set_aspect("equal")

fname = fnames[0]
im2 = ax2.imshow(Image.open(fname)*mask)
data = data1s[0]
qv1 = ax2.quiver(data[:,0], data[:,1], data[:,2], data[:,3], color="g", angles='xy', alpha=0.7, scale_units='xy', scale=0.1)
data = data2s[0]
qv2 = ax2.quiver(data[:,0], data[:,1], data[:,2], data[:,3], color="r", angles='xy', alpha=0.7, scale_units='xy', scale=0.1)

def handler(ev):
    if ev.inaxes==ax1:
        i = argmin( hypot(xs-ev.xdata, ys-ev.ydata))
        #l1.set_data([xs[i]],[ys[i]])
        fname = fnames[i]
        im2.set_data(Image.open(fname)*mask)
        data = data1s[i]
        qv1.set_UVC(data[:,2], data[:,3])
        data = data2s[i]
        qv2.set_UVC(data[:,2], data[:,3])

        #ax1.draw_artist(l1)
        ax2.draw_artist(im2)
        ax2.draw_artist(qv1)
        ax2.draw_artist(qv2)
        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)
 
fig.canvas.mpl_connect('motion_notify_event', handler)
fig.tight_layout()
show()
