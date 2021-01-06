import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

part = sys.argv[1]

with open("out-graph-load-tiff-float.json","r") as fh:
    dataf = json.load(fh)
with open("out-graph-load-tiff-double.json","r") as fh:
    data = json.load(fh)



fig, ax = plt.subplots(figsize=(12,6))


yerr = []
data = data["7"]

lab = ["double", "float"]
rend = 80
for x in range(2):
    for roi_size in range(20, rend, 30):
        if roi_size == 80:
            continue
        xdata = []
        ydata = []
        for size in data[str(roi_size)]:
            parts = data[str(roi_size)][size]
            xdata.append(size)
            #ydata.append(parts["R2C"]["mean"] + parts["C2R"]["mean"] + parts["Multiply"]["mean"])
            #yerr.append(parts["R2C"]["stdev"] + parts["C2R"]["stdev"] + parts["Multiply"]["stdev"])
            ydata.append(parts[part]["mean"])
            yerr.append(parts[part]["stdev"])
        ax.plot(xdata, ydata, label=lab[x] + ', S = {x}'.format(x=roi_size))
    data = dataf["7"]
    rend = 112

#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue') #yerr=yerr)


i = 0
for label in ax.xaxis.get_ticklabels():
    if i%5 != 0:
        label.set_visible(False)
    i += 1

#ax.plot([600*600, 873*873, 1200*1200], [12.0578, 18.7255, 32.2621], label='LINE')

xend = 93
plt.hlines(12.0578, 0, xend, colors='black', label='Image loading')
plt.hlines(18.7255, 0, xend, colors='black', label='')
plt.hlines(32.2621, 0, xend, colors='black', label='')

plt.text(0, 12.5, '600x600', fontsize=12)
plt.text(0, 19.3, '873x873', fontsize=12)
plt.text(0, 32.7, '1200x1200', fontsize=12)

# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

ax.set_title('Duration of the image loading in comparison with GPU processing stage\nfor different input sizes')
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms)')
plt.legend()
# display the plot
plt.show()
fig.tight_layout()
fig.savefig('load-plot.pdf')