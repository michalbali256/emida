#!/usr/bin/python3

def run_python(dset, output, fit_size=3):
    import time
    from PIL import Image
    import numpy as np
    from scipy.signal import correlate, choose_conv_method

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

    started = time.time()
    s = dset.roi.size
    ref = np.asarray(Image.open(dset.ref))
    window = np.hanning(2*s)
    ref_rois = []
    for j,i in dset.roi.positions:
        a = ref[i-s:i+s, j-s:j+s]
        a = a - a.mean()
        a *= window[:,None]
        a *= window[None,:]
        ref_rois.append(a)

    method, times = choose_conv_method(ref_rois[0], ref_rois[0], mode='full', measure=True)
    print("conv method", method, times)
    with open(output, "w") as fh:
        for x,y,fname in dset:
            print(".", end="", flush=True)
            fh.write("{:.6f} {:.6f} {} {}\n".format(x, y, len(dset.roi.positions), fname))
            data = np.asarray(Image.open(fname))
            for (j,i), a in zip(dset.roi.positions, ref_rois):
                b = data[i-s:i+s, j-s:j+s]
                b = b - b.mean()
                b *= window[:,None]
                b *= window[None,:]
                cor = correlate(a, b, mode='full', method=method)
                (yp, xp), q = subpixel_peak(cor, fit_size)
                xp = xp-2*s+1
                yp = yp-2*s+1
                fh.write("{} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(j, i, xp, yp, *q))
    print()
    print(time.time()-started)


def run_emida(dset, output, fit_size=3):
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
        "-p", "{},{}".format(*isize),
        "-c", "{},{}".format(c,c),
        "-f", "{}".format(2*fit_size+1),
        "-q",
    ]
    print(" ".join(args))

    import subprocess, time
    started = time.time()
    with open(output, "w") as fh:
        subprocess.call(args, stdout=fh)

    print(time.time()-started)

from tools import HexDataSet, ROIs
deformed = HexDataSet("../../Testing data/FeAl/DEFORMED_FeAl/DEFORMED_x{x:d}y{y:d}.tif", 7000, 60,
        ref="../../Testing data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif",
        roi=ROIs.load("roi-cryst.txt"),
        ang="../../Testing data/FeAl/DEFORMED_FeAl.ang")

initial = HexDataSet("../../Testing data/FeAl/INITIAL_FeAl/INITIAL_x{x:d}y{y:d}.tif", 7000, 60,
        ref="../../Testing data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif",
        roi=ROIs.load("roi-cryst.txt"),
        ang="../../Testing data/FeAl/INITIAL_FeAl.ang")

if __name__ == "__main__":
    #from pylab import imshow, show
    #imshow(Image.open(deformed.ref))
    #deformed.roi.plot()
    #show()

    #run_emida(deformed.decimate(10), "out-deformed-emida-10.txt", fit_size=3)
    run_python(deformed.decimate(10), "out-deformed-jove-10.txt", fit_size=3)
    #run_emida(deformed.decimate(5), "out-deformed-emida-5.txt", fit_size=3)
    #run_python(deformed.decimate(5), "out-deformed-jove-5.txt", fit_size=3)
    #run_emida(deformed, "out-deformed-emida.txt", fit_size=3)
    #run_python(deformed, "out-deformed-jove.txt", fit_size=3)

    #run_emida(initial.decimate(10), "out-initial-emida-10.txt", fit_size=3)
    #run_python(initial.decimate(10), "out-initial-jove-10.txt", fit_size=3)
    #run_emida(initial.decimate(5), "out-initial-emida-5.txt", fit_size=3)
    #run_python(initial.decimate(5), "out-initial-jove-5.txt", fit_size=3)
    #run_emida(initial, "out-initial-emida.txt", fit_size=3)
    #run_python(initial, "out-initial-jove.txt", fit_size=3)
