import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import json
import sys

with open("TMP.json","r") as fh:
    data = json.load(fh)

fig, ax = plt.subplots(figsize=(5, 5))


yerr = []

labels = []
xdata = []
#data = data["7"]["90"]["50"]
#with open("TMP.json","w") as fh:
#    json.dump(data, fh, indent=4)
for label in data:
    labels.append(label)
    xdata.append(data[label]["mean"])
ax.pie(xdata, labels=labels, autopct='%1.1f%%')
#print(xdata)
#print(ydata)
# plot the data

#ax.set_title('Cross correlation')

# display the plot
plt.show()