import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph.json","r") as fh:
    data = json.load(fh)

part = sys.argv[1]

fig, ax = plt.subplots(figsize=(8,4))


yerr = []

sizes = [(20,20),
(20,80),
(50,50),
(50,80),
(80,80)]

for s in sizes:
    size, roi_size = s
    xdata = []
    ydata = []
    for batch_size in data:
        #print(data[batch_size]["30"]["30"])
        parts = data[batch_size][str(roi_size)][str(size)]
        xdata.append(batch_size)
        #ydata.append(parts[part]["mean"])
        ydata.append(parts[part]["mean"])
        yerr.append(parts[part]["stdev%"])
    print(min(ydata) / max(ydata))
    print(np.argmin(ydata))
    print(np.mean(yerr))
    print()
    ax.plot(xdata, ydata, label='size = {}, S = {}'.format(size, roi_size))

#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue', yerr=yerr)

# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

ax.set_title('The impact of the number of loaded numbers per thread on the sums computation')
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms)')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y')
# display the plot
plt.show()
fig.savefig('batch-plot.pdf')