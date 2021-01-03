import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

part = sys.argv[1]
#with open("out-graph-load-tiff-float.json","r") as fh:
with open("out-graph-loadworkers-float.json","r") as fh:
    dataf = json.load(fh)
#with open("out-graph-loadworkers-double.json","r") as fh:
#    data = json.load(fh)

parrts = ["Load tiff", "Offset", "Offset wait"]
titles = ["The average time needed to load one batch of images for\ndifferent number of load workers (float, batch = 7)",
    "Offset",
    "The average time that the GPU stage of pipeline\nwaits for next loaded pattern (float, batch = 7)"]
fnames = ["loadworkers-load-tiff.pdf", "loadworkers-offset.pdf", "loadworkers-offset-wait.pdf"]

sizes = [(20,20),
(20,80),
(50,20),
(50,50),
(50,80),
(80,50),
(80,110)]




data = dataf
lab = ["double", "float"]
rend = 80
for g in range(3):
    fig, ax = plt.subplots(figsize=(8,4))
    for s in sizes:
        size, roi_size = s
        xdata = []
        ydata = []
        for lw in data:
            parts = data[lw][str(roi_size)][str(size)]
            xdata.append(lw)
            ydata.append(parts[parrts[g]]["mean"])
        ax.plot(xdata, ydata, label='size = {a}, S = {S}'.format(S=roi_size, a=size))
    data = dataf
    ax.set_ylim(ymin=0)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    ax.set_title(titles[g])
    ax.set_xlabel('Number of load workers')
    ax.set_ylabel('time (ms)')
        

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # display the plot
    plt.show()
    fig.savefig(fnames[g])