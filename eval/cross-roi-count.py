import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph-roi-count.json","r") as fh:
    data = json.load(fh)

part = sys.argv[1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


yerr = []
data = data["7"]
for size in range(20,113,10):
    xdata = []
    ydata = []
    for roi_size in data:
        parts = data[roi_size][str(size)]
        xdata.append(roi_size)
        ydata.append(parts[part]["mean"])
        yerr.append(parts[part]["stdev"])
    ax.plot(xdata, ydata, label='size = {x}'.format(x=size))
#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue', yerr=yerr)

# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

# the x coords of this transformation are data, and the

ax.set_title('Performance of Fourier transform with different\nsizes and number of subregions')
ax.set_xlabel('number of subregions')
ax.set_ylabel('time (ms)')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# y coord are axes
# display the plot
plt.show()
fig.savefig('fourier-transform-roi-count.pdf')