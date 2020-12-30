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

for size in range(25, 90, 5):
    xdata = []
    ydata = []
    for batch_size in data:
        #print(data[batch_size]["30"]["30"])
        parts = data[batch_size]["90"][str(size)]
        xdata.append(batch_size)
        ydata.append(parts[part]["mean"]/float(batch_size))
    ax.plot(xdata, ydata, label=str(size))

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