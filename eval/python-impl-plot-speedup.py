import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph-python-impl.json","r") as fh:
    datap = json.load(fh)

with open("out-graph-TOTAL.json","r") as fh:
    data = json.load(fh)

fig, ax = plt.subplots(figsize=(8,5))

data = data["7"]

for roi_size in datap:
    roi_data = data[roi_size]
    xdata = []
    ydata = []
    for size in datap[roi_size]:
        parts = roi_data[size]
        xdata.append(size)
        ydata.append((datap[roi_size][size]*1000)/(parts["TOTAL"]["mean"] + parts["initialization"]["mean"]))
    ax.plot(xdata, ydata, label='S = {x}'.format(x=roi_size))
#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue', yerr=yerr)

# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

ax.set_title('The speedup of GPU implementation compared to python')
ax.set_xlabel('size of subregion')
ax.set_ylabel('speedup')
plt.legend()
# display the plot
plt.show()
fig.savefig('python-impl-plot-speedup.pdf')