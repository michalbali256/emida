#!/usr/bin/python3


class Cursor:
    def __init__(self, pos):
        self.pos = pos

    def init(self, viewer):
        self.ax = viewer.ax1
        x, y = self.pos[0]
        self.l, = self.ax.plot([x], [y], "r+", ms=7, animated=True)

    def update(self, i):
        x, y = self.pos[i]
        self.l.set_data([x],[y])
        self.ax.draw_artist(self.l)

class Dots:
    def __init__(self, pos):
        self.pos = pos

    def init(self, viewer):
        self.ax = viewer.ax1
        self.ax.plot(self.pos[:,0], self.pos[:,1], "r,")

    def update(self, i):
        pass


class Img:
    def __init__(self, fnames):
        self.fnames = fnames

    def init(self, viewer):
        self.ax = viewer.ax2
        fn = self.fnames[0]
        print(fn)
        self.im = self.ax.imshow(self.load(fn), cmap='gray', animated=True)

    def load(self, fn):
        from PIL import Image
        return Image.open(fn)

    def update(self, i):
        fn = self.fnames[i]
        print(fn)
        self.im.set_data(self.load(fn))
        self.ax.draw_artist(self.im)


class HexBg:
    def __init__(self, pos, val, **kw):
        self.pos = pos
        self.val = val
        self.kw = kw

    def init(self, viewer):
        from tools import hexplot
        hexplot(self.pos, self.val, ax=viewer.ax1, **self.kw)

    def update(self, i):
        pass


class Quiver:
    def __init__(self, xy, uv, **kw):
        self.xy = xy
        self.uv = uv
        self.kw = kw

    def init(self, viewer):
        self.ax = viewer.ax2
        xy, uv = self.xy[0], self.uv[0]
        self.q = self.ax.quiver(xy[:,0], xy[:,1], uv[:,0], uv[:,1], animated=True, **self.kw)

    def update(self, i):
        uv = self.uv[i]
        self.q.set_UVC(uv[:,0], uv[:,1])
        self.ax.draw_artist(self.q)

class Viewer:
    def __init__(self, pos, actors):
        from matplotlib.pyplot import subplots
        self.fig, (self.ax1, self.ax2) = subplots(1, 2, figsize=(10,5))
        self.ax1.grid(False)
        self.ax2.grid(False)
        self.fig.tight_layout()

        self._last = 0
        self.pos = pos
        self.actors = actors
        for a in self.actors:
            a.init(self)

        self.fig.canvas.mpl_connect('motion_notify_event', self.motion_handler)
        self.fig.canvas.mpl_connect('draw_event', self.draw_handler)

    def draw_handler(self, ev):
        self._ax1back = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self._ax2back = self.fig.canvas.copy_from_bbox(self.ax2.bbox)

    def motion_handler(self, ev):
        if ev.inaxes != self.ax1:
            return

        from numpy import argmin, hypot
        i = argmin(hypot(self.pos[:,0]-ev.xdata, self.pos[:,1]-ev.ydata))
        if i == self._last:
            return
        self._last = i

        self.fig.canvas.restore_region(self._ax1back)
        self.fig.canvas.restore_region(self._ax2back)
        for a in self.actors:
            a.update(i)
        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)

from matplotlib.pyplot import show

if __name__ == "__main__":
    from run import initial

    dset = initial.decimate(5)
    bg = dset.load_ang()

    v = Viewer(pos=dset.pos, actors=[
        HexBg(bg[:,:2], bg[:,2], cmap='gray'),
        Cursor(dset.pos), 
        Img(dset.fnames)
    ])
   
    data = dset.load_result("out-initial-jove-5.txt")

    v2 = Viewer(pos=dset.pos, actors=[
        HexBg(bg[:,:2], bg[:,2], cmap='gray'),
        Dots(dset.pos),
        Cursor(dset.pos), 
        Img(dset.fnames),
        Quiver(data[:,:,:2], data[:,:,2:4], color="r", angles='xy', scale_units='xy', scale=0.1)
    ])

    show()
