#!/usr/bin/python3

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
    #from matplotlib.pyplot import imshow, show
    #from PIL import Image
    #imshow(Image.open(deformed.ref))
    #deformed.roi.plot()
    #show()

    #deformed.decimate(10).run_gpu("out-deformed-emida-10.txt", fit_size=3)
    #deformed.decimate(10).run_python("out-deformed-jove-10.txt", fit_size=3)
    #deformed.decimate(5).run_gpu("out-deformed-emida-5.txt", fit_size=3)
    deformed.decimate(5).run_python("out-deformed-jove-5.dat", fit_size=3)
    #deformed.run_gpu("out-deformed-emida.txt", fit_size=3)
    #deformed.run_python("out-deformed-jove.txt", fit_size=3)

    #initial.decimate(10).run_gpu("out-initial-emida-10.txt", fit_size=3)
    #initial.decimate(10).run_python("out-initial-jove-10.txt", fit_size=3)
    #initial.decimate(5).run_gpu("out-initial-emida-5.txt", fit_size=3)
    #initial.decimate(5).run_python("out-initial-jove-5.txt", fit_size=3)
    #initial.run_gpu("out-initial-emida.txt", fit_size=3)
    #initial.run_python("out-initial-jove.txt", fit_size=3)
