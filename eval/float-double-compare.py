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
yerrp = []
data = data["7"]

speedup = []

lab = ["double", "float"]
for x in range(2):
    for roi_size in range(20, 112, 30):
        xdata = []
        ydata = []
        for size in data[str(roi_size)]:
            parts = data[str(roi_size)][size]
            xdata.append(size)
            ydata.append(parts[part]["mean"])
            yerr.append(parts[part]["stdev"])
            yerrp.append(parts[part]["stdev%"])
            if x == 0:
                speedup.append(parts[part]["mean"] / dataf["7"][str(roi_size)][size][part]["mean"])
        ax.plot(xdata, ydata, label=lab[x] + ', S = {x}'.format(x=roi_size))
        
    data = dataf["7"]

print(np.mean(yerrp))
print(np.min(speedup))
print(np.max(speedup))
print(np.mean(speedup))
i = 0
for label in ax.xaxis.get_ticklabels():
    if i%5 != 0:
        label.set_visible(False)
    i += 1

#ax.plot([600*600, 873*873, 1200*1200], [12.0578, 18.7255, 32.2621], label='LINE')
"""
xend = 93
plt.hlines(12.0578, 0, xend, colors='black', label='Image loading')
plt.hlines(18.7255, 0, xend, colors='black', label='')
plt.hlines(32.2621, 0, xend, colors='black', label='')

plt.text(0, 12.5, '600x600', fontsize=12)
plt.text(0, 19.3, '873x873', fontsize=12)
plt.text(0, 32.7, '1200x1200', fontsize=12)
"""

ax.set_title('The performance of the GPU stage with different floating precision, size and number of subregions')
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms)')
plt.legend()
# display the plot
plt.show()
fig.tight_layout()
fig.savefig('float-double-compare.pdf')