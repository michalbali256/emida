#!/usr/bin/python3

from math import sqrt
def hex_pos(size, step):
    m = int(size/step/sqrt(0.75))
    n = int(size/step)
    for j in range(m+1):
        for i in range(n-j%2+1):
            x = (i+j%2/2)*step
            y = j*sqrt(0.75)*step
            yield x, y

from scipy.signal import correlate
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


import numpy as np
def read_roi(fname):
    with open(fname) as fh:
        s = np.loadtxt(fh, dtype=int, max_rows=1)
        pos = np.loadtxt(fh,  dtype=int, ndmin=2)
    return s, pos

if __name__ == "__main__":

    ref = "../../Testing data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif"
    fmt = "../../Testing data/FeAl/DEFORMED_FeAl/DEFORMED_x{x:d}y{y:d}.tif"
    #fmt = "Testing data/FeAl/INITIAL_FeAl/INITIAL_x{x:d}y{y:d}.tif"
    roi = "roi-cryst.txt"
    #roi = "roi-test.txt"
    fit_size = 3

    #work = "test.txt"
    # minimal step is 60
    step = 300

    work = "def-{}.txt".format(step)
    it = hex_pos(7000, step)
    with open(work, "w") as fh:
        for x,y in it: 
            print(x, y, fmt.format(x=int(x), y=int(y)), file=fh)

    import time
    if 1:
        import subprocess
        s, pos = read_roi(roi)
        c = 4*s-1
        #c = 4*s//3-1
        started = time.time()
        with open("out-emida-{}.txt".format(step),"w") as fh:
            args = [
                "../build/bin/emida",
                "-d", work,
                "-i", ref,
                "-b", roi,
                "-p", "873,873",
                "-c", "{},{}".format(c,c),
                "-f", str(2*fit_size+1),
                "-q",
            ]
            print(" ".join(args))
            subprocess.call(args, stdout=fh)
        print(time.time()-started)

    if 1:
        import numpy as np
        from PIL import Image
        s,positions = read_roi(roi)
        ref = np.asarray(Image.open(ref))
        window = np.hanning(2*s)
        ref_rois = []
        for j,i in positions:
            a = ref[i-s:i+s, j-s:j+s]
            a = a - a.mean()
            a *= window[:,None]
            a *= window[None,:]
            ref_rois.append(a)
        started = time.time()
        with open("out-jove-{}.txt".format(step),"w") as fh:
            for l in open(work):
                print(".", end="", flush=True)
                x, y, fname = l.rstrip().split(maxsplit=2)
                x, y = float(x), float(y)
                fh.write("{:.6f} {:.6f} {} {}\n".format(x, y, len(positions), fname))
                data = np.asarray(Image.open(fname))
                for (j,i), a in zip(positions, ref_rois):
                    b = data[i-s:i+s, j-s:j+s]
                    b = b - b.mean()
                    b *= window[:,None]
                    b *= window[None,:]
                    cor = correlate(a, b, mode='full')
                    (yp, xp), q = subpixel_peak(cor, fit_size)
                    xp = xp-2*s+1
                    yp = yp-2*s+1
                    fh.write("{} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(j, i, xp, yp, *q))
        print()
        print(time.time()-started)
