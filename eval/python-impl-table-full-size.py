import numpy as np
import json
import sys

with open("out-graph-full-size.json","r") as fh:
    data = json.load(fh)

sizes = [(20,20),
(20,110),
(50,50),
(50,80),
(80,50),
(110,20),
(110,110)]

for s in sizes:
    size, roi_size  = s
    #print('size = {}, S = {}'.format(size, roi_size), '& ', end='')
    print(size, '&', roi_size, '& ', end='')
    
    for label in data:
        val = data[label][str(roi_size)][str(size)]
        print("{:.1f}".format(val), '&', end=' ')
        if label != 'python':
            speedup = data["python"][str(roi_size)][str(size)] / data[label][str(roi_size)][str(size)]
            print("{:.1f}x".format(speedup), '&', end=' ')
        
    print(' \\\\')
