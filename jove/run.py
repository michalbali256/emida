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
    yp, xp = np.unravel_index(np.argmax(d), d.shape)

    #from pylab import matshow, show, plot
    #matshow(d)
    #plot([xp],[yp],"x")
    #show()

    y, x = yp-s, xp-s
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
        xs, ys = -float("nan"), -float("nan")
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

    #work = "test.txt"
    work = "def.txt"
    it = hex_pos(7000, 600) # minimal step is 60
    with open(work, "w") as fh:
        for x,y in it: 
            print(x, y, fmt.format(x=int(x), y=int(y)), file=fh)

    import time
    if 1:
        import subprocess
        s, pos = read_roi(roi)
        np.savetxt("roi.emida", pos-s, fmt="%d", delimiter=",")
        started = time.time()
        with open("out-emida.txt","w") as fh:
            subprocess.call([
                "../build/bin/emida",
                "-d", work,
                "-i", ref,
                "-b", roi,
                "-p", "873,873",
                "-r", "0,0,0,0", # useles but required
                #"-c", "25,25",
                "-c", "{},{}".format(4*s-1,4*s-1),
                "-s", "{},{}".format(2*s,2*s),
            ], stdout=fh)
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
        with open("out-jove.txt","w") as fh:
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
                    (yp, xp), q = subpixel_peak(cor)
                    xp = xp-2*s+1
                    yp = yp-2*s+1
                    fh.write("{} {} {:.6f} {:.6f}\n".format(j,i,xp,yp))
        print()
        print(time.time()-started)
