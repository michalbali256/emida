#!/usr/bin/python3

from pylab import *

from PIL import Image
fname = "../../Testing data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif"
img = asarray(Image.open(fname))
mask = asarray(Image.open("mask.png"))==255

imshow(img*mask, cmap="gray")
axis("image")
grid(False)

from run import read_roi
s, positions = read_roi(sys.argv[1])

from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import PatchCollection
areas = [ Rectangle( (j-s, i-s), 2*s, 2*s) for j,i in positions ]
gca().add_collection(PatchCollection(areas, facecolor='none', edgecolor='b'))
show()
