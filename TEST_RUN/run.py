#!/usr/bin/python3

from math import sqrt
import numpy as np
import time
import subprocess

def hex_pos(size, step):
    m = int(size/step/sqrt(0.75))
    n = int(size/step)
    for j in range(m+1):
        for i in range(n-j%2+1):
            x = (i+j%2/2)*step
            y = j*sqrt(0.75)*step
            yield x, y

if __name__ == "__main__":

    ref = "../test_data/initial/INITIAL_x0y0.tif"
    fmt = "../test_data/deformed/DEFORMED_x{x:d}y{y:d}.tif"
    
    roi = "roi-cryst.txt"

    work = "def.txt"
    #set the size of processed rectangle here
    it = hex_pos(1200, 60)
    #it = hex_pos(7000, 60) #the size of original data
    with open(work, "w") as fh:
        for x,y in it: 
            print(x, y, fmt.format(x=int(x), y=int(y)), file=fh)
    
    executable_path = "../build/x64-Release/bin/emida"
    #executable_path = "../build/bin/emida"
    
    started = time.time()
    with open("out-emida.txt","w") as fh:
        subprocess.call([
            "../build/x64-Release/bin/emida",
            "-d", work,
            "-i", ref,
            "-b", roi,
            "--batchsize", "7",
            "--crosspolicy", "fft",
            "--precision","float",
            "--loadworkers", "5",
            "-f","3",
            "-a"
        ], stdout=fh)
    print(time.time()-started)
