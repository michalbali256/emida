import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph-compare-brute.json","r") as fh:
    data_brute = json.load(fh)
with open("out-graph-compare-fft.json","r") as fh:
    data_fft = json.load(fh)

fig, ax = plt.subplots(figsize=(8,4))


yerr = []
data = data_brute["7"]
for roi_size in data:
    roi_data = data[roi_size]
    xdata = []
    ydata = []
    for size in roi_data:
        parts = roi_data[size]
        xdata.append(size)
        ydata.append(parts["Run cross corr"]["mean"])
    ax.plot(xdata, ydata, label='Definition-based, S = {x}'.format(x=roi_size), dashes=[6, 2])
    
data = data_fft["7"]
for roi_size in data:
    roi_data = data[roi_size]
    xdata = []
    ydata = []
    for size in roi_data:
        parts = roi_data[size]
        xdata.append(size)
        ydata.append(parts["R2C"]["mean"] + parts["C2R"]["mean"] + parts["Multiply"]["mean"])
    ax.plot(xdata, ydata, label='FFT, S = {x}'.format(x=roi_size))

i = 0
for label in ax.xaxis.get_ticklabels():
    if i%2 != 0:
        label.set_visible(False)
    i += 1


ax.set_title('FFT implementation versus definition--based implementation')
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms)')

plt.legend()
# display the plot
plt.show()
fig.savefig('cross-compare.pdf')