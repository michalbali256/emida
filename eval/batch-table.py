import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph-batch-double.json","r") as fh:
    data = json.load(fh)

part = sys.argv[1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


yerr = []



for roi_size in data["1"]:
    print(roi_size, '&', end=' ')
    for size in data["1"]["20"]:
        xdata = []
        ydata = []
        for batch_size in data:
            #print(data[batch_size]["30"]["30"])
            parts = data[batch_size][roi_size][str(size)]
            xdata.append(batch_size)
            #ydata.append(parts[part]["mean"])
            ydata.append(parts[part]["mean"]/float(batch_size))
            yerr.append(parts[part]["stdev%"])
            
        impr = (1- min(ydata) / max(ydata)) * 100
        print('{:.1f}\\%'.format(impr), "&", np.argmin(ydata), "&", end=' ')
        
        ax.plot(xdata, ydata, label='S = {x}'.format(x=roi_size))
    print("\\\\")

#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue', yerr=yerr)

# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

ax.set_title('The impact of different values of batch parameter')
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms) per one batch')
plt.legend()
# display the plot
plt.show()
fig.tight_layout()
fig.savefig('batch-plot.pdf')