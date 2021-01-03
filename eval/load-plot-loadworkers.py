import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

part = sys.argv[1]
#with open("out-graph-load-tiff-float.json","r") as fh:
with open("out-graph-loadworkers-float.json","r") as fh:
    dataf = json.load(fh)
with open("out-graph-loadworkers-double.json","r") as fh:
    data = json.load(fh)

fig, ax = plt.subplots(figsize=(12,6))

data = dataf
lab = ["double", "float"]
rend = 80
for x in range(1):
    for roi_size in range(20, 110, 30):
        for size in range(20, 110, 30):
            xdata = []
            ydata = []
            for lw in data:
                parts = data[lw][str(roi_size)][str(size)]
                xdata.append(lw)
                ydata.append(parts[part]["mean"])
            ax.plot(xdata, ydata, label=lab[x] + ', size = {a}, S = {S}'.format(S=roi_size, a=size))
    data = dataf
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