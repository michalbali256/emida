import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("out-graph-full-size.json","r") as fh:
    data = json.load(fh)

fig, ax = plt.subplots(figsize=(8,5))

sizes = [(20,20),
(50,50),
(50,80),
(80,50),
(110,110)]

labels = []
for s in sizes:
    roi_size, size = s
    labels.append('size = {}, S = {}'.format(size, roi_size))

width = 0.35  # the width of the bars
width_corr = -1
x = np.arange(len(labels))*2  # the label locations

for label in data:
    ydata = []
    for s in sizes:
        roi_size, size = s
        ydata.append(int(data[label][str(roi_size)][str(size)]))
    
    rect = ax.bar(x + width*width_corr, ydata, width, label=label)
    
    rects.append(rect)
    width_corr += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Comparison of python and GPU implementation for full dataset')
ax.set_yscale("log")
ax.set_ylabel('time (s)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for rect in rects:
    autolabel(rect)

fig.tight_layout()


plt.legend()
# display the plot
plt.show()
fig.savefig('python-impl-plot-speedup.pdf')