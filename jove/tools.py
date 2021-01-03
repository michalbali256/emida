#!/usr/bin/python3

import numpy as np
from PIL import Image

def hexiter(size, step):
    m = int(size/step/np.sqrt(0.75))
    n = int(size/step)
    for j in range(m+1):
        for i in range(n-j%2+1):
            x = (i+j%2/2)*step
            y = j*np.sqrt(0.75)*step
            yield x, y

from hexplot import hexplot1 as hexplot

def subpixel_peak(d, s=1):
    """Fit polynomial quadratic in x and y to the neighbourhood of maximum,
       to determine maximum with subpixel precision"""
    y, x = np.unravel_index(np.argmax(d[s:d.shape[0]-s,s:d.shape[1]-s]), (d.shape[0]-2*s, d.shape[1]-2*s))

    if d[y+s, x+s] == d[d.shape[0]//2, d.shape[1]//2]:
        # default to center, if not worse
        y, x = d.shape[0]//2-s, d.shape[1]//2-s

    #from pylab import matshow, show, plot
    #matshow(d)
    #plot([xp],[yp],"x")
    #show()

    l = d[y:y+2*s+1, x:x+2*s+1]
    yi, xi = np.indices(l.shape)
    i = np.ones(l.shape)
    A = np.dstack([i, xi, yi, xi*xi, xi*yi, yi*yi]).reshape((-1,6))
    b = l.reshape(-1)
    #print(repr(b))
    q = np.linalg.lstsq(A, b, rcond=None)[0]
    #print(repr(q))
    try:
        xs, ys = np.linalg.solve([[2*q[3],   q[4]],
                                  [  q[4], 2*q[5]]], [-q[1], -q[2]])
    except np.linalg.LinAlgError:
        xs, ys = s, s # if subpixel failed default to integer maximum

    return (y+ys, x+xs), q


from scipy.signal import choose_conv_method, correlate
from scipy import fft
class DataSet:
    def __init__(self, fmt, size, step, ang=None, ref=None, roi=None):
        self.fmt = fmt
        self.size = size
        self.step = step
        self.ang = ang
        self.ref = ref
        self.roi = roi

        self.pos = np.array([ (x,y) for x, y, _ in self])
        self.fnames = [f for _, _, f in self]

    def decimate(self, n):
        return self.__class__(self.fmt, self.size, self.step*n, ang=self.ang, ref=self.ref, roi=self.roi)

    def __iter__(self):
        for x, y in self.iter(self.size, self.step):
            yield x/100, y/100, self.fmt.format(x=int(x), y=int(y))

    def roi_iter(self, data, flip=False):
        s = self.roi.size
        for j,i in self.roi.positions:
            r = data[i-s:i+s, j-s:j+s]
            r = r - r.mean()
            r *= self.window[:,None]
            r *= self.window[None,:]
            n = np.sqrt((r*r).sum())
            if n > 0:
                r /= n # normalize
            if flip:
                r = r[::-1,::-1]
            yield fft.rfft2(r, (4*s,4*s))

    def get_ref(self):
        s = self.roi.size
        self.window = np.hanning(2*s)

        ref = np.asarray(Image.open(self.ref))
        self.ref_rois = list( self.roi_iter(ref, flip=False) )

    def get_cor(self, fname, fit_size=3):
        s = self.roi.size
        data = np.asarray(Image.open(fname))
        for a, b in zip(self.ref_rois, self.roi_iter(data, flip=True)):
            cor = fft.irfft2(a*b, (4*s, 4*s))[:-1,:-1]
            (yp, xp), q = subpixel_peak(cor, fit_size)
            xp = xp-2*s+1
            yp = yp-2*s+1
            yield cor, xp, yp, q

    def run_python(self, output, fit_size=3):
        import time
        import multiprocessing
        nthreads = multiprocessing.cpu_count()
        with fft.set_workers(nthreads):
            started = time.time()
            self.get_ref()
            if output.endswith(".txt"):
                with open(output, "w") as fh:
                    for x, y, fname in self:
                        print(".", end="", flush=True)
                        fh.write("{:.6f} {:.6f} {} {}\n".format(x, y, len(self.roi.positions), fname))
                        for (j,i), (cor, xp, yp, q) in zip(self.roi.positions, self.get_cor(fname, fit_size=fit_size)):
                            fh.write("{} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(j, i, xp, yp, *q))
            else:
                with open(output, "wb") as fh:
                    for x, y, fname in self:
                        print(".", end="", flush=True)
                        data = np.asarray([ (j,i,xp,yp,*q) for (j,i), (cor, xp, yp, q) in zip(self.roi.positions, self.get_cor(fname, fit_size=fit_size)) ], dtype=float)
                        data.tofile(fh)
            print()
            print(time.time()-started)

    def run_gpu(dset, output, fit_size=3):
        with open("emida-work.txt", "w") as fh:
            for x, y, fn in dset:
                print(x, y, fn, file=fh)

        dset.roi.save("emida-roi.txt")

        isize = Image.open(dset.ref).size

        c = 4*dset.roi.size-1
        #c = 4*dset.roi.size//3-1

        args = [
            "../build/bin/emida",
            "-d", "emida-work.txt",
            "-b", "emida-roi.txt",
            "-i", dset.ref,
            #"-c", "{},{}".format(c,c),
            "-f", "{}".format(2*fit_size+1),
            "-q",
            "--crosspolicy=fft"
        ]
        print(" ".join(args))

        import subprocess, time
        started = time.time()
        with open(output, "w") as fh:
            subprocess.call(args, stdout=fh)

        print(time.time()-started)

    def load_ang(self):
        return np.loadtxt(self.ang, usecols=(3,4,5))

    def _load_result_txt(self, fname):
        pos = []
        fnames = []
        data = []
        import time
        started = time.time()
        with open(fname) as fh:
            while True:
                print(".", end="", flush=True)
                line = fh.readline()
                if not line:
                    break
                x, y, n, fn = line.rstrip().split(maxsplit=3)
                x, y, n = float(x), float(y), int(n)
                pos.append((x,y))
                fnames.append(fn)
                data.append(np.loadtxt(fh, max_rows=n))
        pos = np.asarray(pos)
        data = np.asarray(data)
        print("loaded in", time.time()-started, "s")
        return pos, fnames, data

    def load_result(self, fname):
        txt = fname[:-4]+".txt"
        dat = fname[:-4]+".dat"

        import os.path
        if os.path.exists(dat):
            import time
            started = time.time()
            data = np.fromfile(dat, dtype=float).reshape(len(self.fnames), len(self.roi.positions), 2+2+6)
            print("loaded in", time.time()-started, "s")
            return data

        elif os.path.exists(txt):
            pos, fnames, data = self._load_result_txt(txt)
            assert np.allclose(pos, self.pos)
            assert fnames == self.fnames
            data.tofile(dat)
            return data

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
            positions = np.loadtxt(fh,  dtype=int, ndmin=2)
        return cls(size, positions)

    def save(self, fname):
        with open(fname, "w") as fh:
            print(self.size, file=fh)
            np.savetxt(fh, self.positions, fmt="%d")

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

if __name__ == "__main__":

    data = HexDataSet("../../Testing data/FeAl/DEFORMED_FeAl/DEFORMED_x{x:d}y{y:d}.tif", 7000, 60,
        ang="../../Testing data/FeAl/DEFORMED_FeAl.ang")
    for x,y,f in data.decimate(20):
        print(x,y,f)

    from matplotlib.pyplot import imshow, show, savefig
    #val = np.random.poisson(500*np.exp( -((data.pos-35)**2).sum(axis=-1)/15/15/2))
    val = data.load_ang()[:,2]
    data.plot(data.pos, val)
    #savefig("test.png", dpi=500)
    show()

    mask = np.asarray(Image.open("mask.png"))==255
    imshow(mask)
    rois = ROIs.regular(48, 64, mask)
    #rois = ROIs.load("roi-cryst.txt")
    rois.plot()
    rois.save("/dev/stdout")
    show()
