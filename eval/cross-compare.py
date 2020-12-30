import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph-compare-brute.json","r") as fh:
    data_brute = json.load(fh)
with open("out-graph-compare-fft.json","r") as fh:
    data_fft = json.load(fh)

part = sys.argv[1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


yerr = []
data = data_brute["7"]
for roi_size in data:
    roi_data = data[roi_size]
    xdata = []
    ydata = []
    for size in roi_data:
        parts = roi_data[size]
        xdata.append(size)
        ydata.append(parts[part]["mean"])
    ax.plot(xdata, ydata)
    
data = data_fft["7"]
for roi_size in data:
    roi_data = data[roi_size]
    xdata = []
    ydata = []
    for size in roi_data:
        parts = roi_data[size]
        xdata.append(size)
        ydata.append(parts["R2C"]["mean"] + parts["C2R"]["mean"] + parts["Multiply"]["mean"])
    ax.plot(xdata, ydata)
#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue', yerr=yerr)

# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

ax.set_title('Cross correlation')

# display the plot
plt.show()