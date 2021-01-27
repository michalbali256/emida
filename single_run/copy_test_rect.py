#!/usr/bin/python3

from math import sqrt
import numpy as np
import time
import shutil
import os

def hex_pos(size, step):
    m = int(size/step/sqrt(0.75))
    n = int(size/step)
    ls = []
    for j in range(m+1):
        for i in range(n-j%2+1):
            x = (i+j%2/2)*step
            y = j*sqrt(0.75)*step
            ls.append((x, y))
    return ls



def read_roi(fname):
    with open(fname) as fh:
        s = np.loadtxt(fh, dtype=int, max_rows=1)
        pos = np.loadtxt(fh,  dtype=int, ndmin=2)
    return s, pos

if __name__ == "__main__":

    ref = "../data/FeAl/INITIAL_FeAl/INITIAL_x0y0.tif"
    fmt = "../data/FeAl/DEFORMED_FeAl/DEFORMED_x{x:d}y{y:d}.tif"
    #fmt = "E:/emida/data/FeAl/DEFORMED_FeAl/DEFORMED_x{x:d}y{y:d}.tif"
    
    dest_ref = "../test_data/initial/INITIAL_x0y0.tif"
    dest_fmt = "../test_data/deformed/DEFORMED_x{x:d}y{y:d}.tif"
    
    
    
    it = hex_pos(1200, 60) # minimal step is 60
    print(len(it))
    #with open(work, "w") as fh:
    #    for x,y in it: 
    #        print(x, y, fmt.format(x=int(x), y=int(y)), file=fh)
    #print(it)
    
    os.makedirs(os.path.dirname(dest_fmt.format(x=1,y=1)), exist_ok=True)
    for i in it:
        x, y = i
        x = int(x)
        y = int(y)
        shutil.copyfile(fmt.format(x=x, y=y), dest_fmt.format(x=x,y=y))
        print('.', end='', flush=True)
    
    os.makedirs(os.path.dirname(dest_ref), exist_ok=True)
    shutil.copyfile(ref, dest_ref)
