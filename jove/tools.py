#!/usr/bin/python3

def hexiter(size, step):
    from math import sqrt
    m = int(size/step/sqrt(0.75))
    n = int(size/step)
    for j in range(m+1):
        for i in range(n-j%2+1):
            x = (i+j%2/2)*step
            y = j*sqrt(0.75)*step
            yield x, y


def hexplot(pos, C, step=None, cmap=None, vmin=None, vmax=None, ax=None, **kwargs):
    if step is None:
        step = pos[1,0]-pos[0,0]

    from numpy import array, sqrt
    s3 = 1/sqrt(3)
    hx = array([(0,  2*s3), ( 1,  s3), ( 1, -s3), 
                (0, -2*s3), (-1, -s3), (-1,  s3)])*step/2
    
    pos6 = pos[:,None,:] + hx[None,:,:]

    kwargs.setdefault('edgecolors', 'none')
    kwargs.setdefault('antialiaseds', False)

    from matplotlib.collections import PolyCollection
    col = PolyCollection(pos6, **kwargs)
    col.set_array(C)
    col.set_cmap(cmap)
    if vmin is not None or vmax is not None:
        col.set_clim(vmin, vmax)
    else:
        col.autoscale_None()

    if ax is None:
        from matplotlib.pyplot import gca
        ax = gca()

    ax.grid(False)

    minx = pos6[...,0].min()
    maxx = pos6[...,0].max()
    miny = pos6[...,1].min()
    maxy = pos6[...,1].max()
    ax.update_datalim([ (minx, miny), (maxx, maxy) ])
    ax.margins(0)
    #ax.autoscale_view()
    ax.add_collection(col)
    ax.set_aspect("equal")
    return col


from numpy import savetxt, loadtxt, asarray, array

class DataSet:
    def __init__(self, fmt, size, step, ang=None, ref=None, roi=None):
        self.fmt = fmt
        self.size = size
        self.step = step
        self.ang = ang
        self.ref = ref
        self.roi = roi

        self.pos = array([ (x,y) for x, y, _ in self])
        self.fnames = [f for _, _, f in self]

    def decimate(self, n):
        return self.__class__(self.fmt, self.size, self.step*n, ang=self.ang, ref=self.ref, roi=self.roi)

    def __iter__(self):
        for x, y in self.iter(self.size, self.step):
            yield x/100, y/100, self.fmt.format(x=int(x), y=int(y))


class HexDataSet(DataSet):
    iter = staticmethod(hexiter)
    plot = staticmethod(hexplot)


class ROIs:
    def __init__(self, size, positions):
        self.size = size
        self.positions = positions

    @classmethod
    def load(cls, fname):
        with open(fname) as fh:
            size = int(fh.readline())
            positions = loadtxt(fh,  dtype=int, ndmin=2)
        return cls(size, positions)

    def save(self, fname):
        with open(fname, "w") as fh:
            print(self.size, file=fh)
            savetxt(fh, self.positions, fmt="%d")

    def plot(self, ax=None):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        s = self.size
        areas = [ Rectangle((j-s, i-s), 2*s, 2*s) for j,i in self.positions ]
        col = PatchCollection(areas, facecolor='none', edgecolor='b')
        if ax is None:
            from matplotlib.pyplot import gca
            ax = gca()
        ax.grid(False)
        ax.add_collection(col)
        return col

    @classmethod
    def regular(cls, size, step, mask):
        positions = [ (i,j) for i in range(size, mask.shape[0]-size+1, step) 
                            for j in range(size, mask.shape[1]-size+1, step) 
                            if mask[i-size:i+size, 
                                    j-size:j+size].all() ]
        return cls(size, positions)

def load_result(fname):
    pos = []
    fnames = []
    data = []
    with open(fname) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            x, y, n, fn = line.rstrip().split(maxsplit=3)
            x, y, n = float(x), float(y), int(n)
            pos.append((x,y))
            fnames.append(fn)
            data.append(loadtxt(fh, max_rows=n))
    pos = asarray(pos)
    data = asarray(data)
    return pos, fnames, data

if __name__ == "__main__":

    data = HexDataSet("../../Testing data/FeAl/DEFORMED_FeAl/DEFORMED_x{x:d}y{y:d}.tif", 7000, 60)
    for x,y,f in data.decimate(20):
        print(x,y,f)
 
    from pylab import array, poisson, exp, show
    val = poisson(500*exp( -((data.pos-35)**2).sum(axis=-1)/15/15/2))
    data.plot(data.pos, val)
    show()

    from PIL import Image
    from pylab import asarray, imshow, show
    mask = asarray(Image.open("mask.png"))==255
    imshow(mask)
    #rois = ROIs.regular(48, 64, mask)
    rois = ROIs.load("roi-cryst.txt")
    rois.plot()
    rois.save("/dev/stdout")
    show()
