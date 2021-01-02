#!/usr/bin/python3

from math import sqrt
import numpy as np

def hex_pos(size, step):
    m = int(size/step/sqrt(0.75))
    n = int(size/step)
    for j in range(m+1):
        for i in range(n-j%2+1):
            x = (i+j%2/2)*step
            y = j*sqrt(0.75)*step
            yield x, y

def read_roi(fname):
    with open(fname) as fh:
        s = np.loadtxt(fh, dtype=int, max_rows=1)
        pos = np.loadtxt(fh,  dtype=int, ndmin=2)
    return s, pos

if __name__ == "__main__":

    ref = "../data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif"
    fmt = "../data/FeAl/DEFORMED_FeAl/DEFORMED_x{x:d}y{y:d}.tif"
    
    dest_size = 600
    dest_ref = "../data/FeAl/INITIAL_{size}/INITIAL_x0y0.tif".format(size=dest_size)
    dest_fmt = "../data/FeAl/DEFORMED_{size}/DEFORMED_x{x:d}y{y:d}.tif"
    
    roi = "roi-cryst.txt"
    #roi = "roi-test.txt"

    #work = "test.txt"
    work = "def.txt"
    it = hex_pos(7000, 600) # minimal step is 60

    
    from PIL import Image
    im = Image.open(ref)
    new_image = im.resize((dest_size, dest_size), Image.NEAREST)
    new_image.save(dest_ref)
    for sur in it:
        x,y = sur
        x=int(x)
        y=int(y)
        im = Image.open(fmt.format(x=x, y=y))
        new_image = im.resize((dest_size, dest_size), Image.NEAREST)
        new_image.save(dest_fmt.format(size=dest_size, x=x, y=y))
        
