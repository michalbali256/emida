import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph-BIG.json","r") as fh:
    data = json.load(fh)

part = sys.argv[1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


yerr = []
data = data["7"]
for size in range(20,90,10):
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

ax.set_title('Cross correlation')
plt.legend()
# display the plot
plt.show()