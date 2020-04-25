#!/usr/bin/python3

from pylab import *
from sim import euler2rot, sim_pattern, poles, M

calib = array([48.8235, 77.5223, 69.8357])
angles = array([2.1, 150.8, 261.5])


from PIL import Image
fname = "../../Testing data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif"
img = array(Image.open(fname))
mask = asarray(Image.open("mask.png"))==255

imshow(img, cmap="gray")
axis("image")
grid(False)

assert img.shape[0] == img.shape[1]

calib = img.shape[0]*(array([0,1,0]) + array([1,-1,1])*calib/100)
#sim_pattern( euler2rot(angles).dot(M), calib, w=0)
sim_pattern( euler2rot(angles).dot(M), calib, w=0.015)
x,y = poles( euler2rot(angles).dot(M), calib, M=7)
plot(x,y,"o")

s = 48
positions = [ (j,i) for i,j in zip(y.astype(int),x.astype(int)) if s < i < img.shape[0]-s and s<j<img.shape[1]-s and mask[i-s:i+s, j-s:j+s].all() ]
positions = list(set(positions)) # unique

import sys
print(s, file=sys.stdout)
savetxt(sys.stdout, positions, fmt="%d")
show()
