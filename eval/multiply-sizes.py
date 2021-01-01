import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

part = sys.argv[1]




fig, ax = plt.subplots(figsize=(12,6))



for roi_size in range(20, 111, 30):
    xdata = []
    ydata = []
    ydatad = []
    for size in range(20, 110):
        
        xdata.append(size)
        #ydata.append(parts["R2C"]["mean"] + parts["C2R"]["mean"] + parts["Multiply"]["mean"])
        #yerr.append(parts["R2C"]["stdev"] + parts["C2R"]["stdev"] + parts["Multiply"]["stdev"])
        act_size = size*4
        ydata.append(4*act_size*act_size*roi_size/1024)
        ydatad.append(8*act_size*act_size*roi_size/1024)
    ax.plot(xdata, ydata, label='S = {x} float'.format(x=roi_size))
    ax.plot(xdata, ydatad, label='S = {x} double'.format(x=roi_size))
#print(xdata)
#print(ydata)
# plot the data

#ax.bar(xdata, ydata, color='tab:blue') #yerr=yerr)

ax.ticklabel_format(useOffset=False, style='plain')


# set the limits
#ax.set_xlim([0, 100])
#ax.set_ylim([0, 10])

ax.set_title('')
ax.set_xlabel('size of subregion')
ax.set_ylabel('bytes')
plt.legend()
# display the plot
plt.show()
fig.tight_layout()
#fig.savefig('ply-plot-{t}.pdf'.format(t=typ))