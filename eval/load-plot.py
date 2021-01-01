import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

part = sys.argv[1]

with open("out-graph-load-tiff-float.json","r") as fh:
#with open("out-graph-float.json","r") as fh:
    data = json.load(fh)



fig, ax = plt.subplots(figsize=(12,6))


yerr = []
data = data["7"]


for roi_size in data:
    xdata = []
    ydata = []
    for size in data[roi_size]:
        parts = data[roi_size][size]
        xdata.append(size)
        #ydata.append(parts["R2C"]["mean"] + parts["C2R"]["mean"] + parts["Multiply"]["mean"])
        #yerr.append(parts["R2C"]["stdev"] + parts["C2R"]["stdev"] + parts["Multiply"]["stdev"])
        ydata.append(parts[part]["mean"])
        yerr.append(parts[part]["stdev"])
    ax.plot(xdata, ydata, label='S = {x}'.format(x=roi_size))
#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue') #yerr=yerr)


i = 0
for label in ax.xaxis.get_ticklabels():
    if i%5 != 0:
        label.set_visible(False)
    i += 1


# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

ax.set_title('Time that the thread computing offsets waits for loading of TIFF')
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms)')
plt.legend()
# display the plot
plt.show()
fig.tight_layout()
fig.savefig('load-plot.pdf')