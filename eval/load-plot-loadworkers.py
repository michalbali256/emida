import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

part = sys.argv[1]
#with open("out-graph-load-tiff-float.json","r") as fh:
with open("out-graph-loadworkers.json","r") as fh:
    data = json.load(fh)

fig, ax = plt.subplots(figsize=(12,6))



for roi_size in data["1"]:
    for size in range(20, 110, 30):
        xdata = []
        ydata = []
        for lw in data:
            parts = data[lw][roi_size][str(size)]
            xdata.append(lw)
            ydata.append(parts[part]["mean"])
        ax.plot(xdata, ydata, label='size = {a}, S = {S}'.format(S=roi_size, a=size))
#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue') #yerr=yerr)
ax.set_ylim(ymin=0)


ax.set_title('Time that the thread computing offsets waits for loading of TIFF')
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms)')
plt.legend()
# display the plot
plt.show()
fig.tight_layout()
fig.savefig('load-plot-loadworkers.pdf')