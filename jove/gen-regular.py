#!/usr/bin/python3

from numpy import asarray, savetxt
from PIL import Image
mask = asarray(Image.open("mask.png"))==255

s = 48
d = 64
positions = [ (i,j) for i in range(s,mask.shape[0]-s+1,d) for j in range(s,mask.shape[1]-s+1,d) if mask[i-s:i+s, j-s:j+s].all() ]

import sys
print(s, file=sys.stdout)
savetxt(sys.stdout, positions, fmt="%d")
