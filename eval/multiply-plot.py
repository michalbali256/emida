import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

part = sys.argv[1]
typ = sys.argv[2]

#with open("out-graph-{t}.json".format(t=typ),"r") as fh:
with open("out-graph-BIG.json","r") as fh:
    data = json.load(fh)



fig, ax = plt.subplots(figsize=(12,6))


yerr = []
yerrp = []
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
        yerrp.append(parts[part]["stdev%"])
    ax.plot(xdata, ydata, label='S = {x}'.format(x=roi_size))

print(np.mean(yerrp))
print(np.mean(yerr))
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

ax.set_title('Performance of complex matrix multiplication - {t} (batch = 7)'.format(t=typ))
ax.set_xlabel('size of subregion')
ax.set_ylabel('time (ms)')
plt.legend()
# display the plot
plt.show()
fig.tight_layout()
fig.savefig('multiply-plot-{t}.pdf'.format(t=typ))