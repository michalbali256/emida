import numpy as np
import json
import sys

with open("out-graph-load-tiff-float.json","r") as fh:
    data = json.load(fh)

data = data["7"]

tiff_size = 873*873
tiff_bytes = tiff_size*2
batch_size = tiff_bytes * 7
batch_size_mb = batch_size / (1024*1024)

print('S\\siz', end="&")
for x in range(20, 112, 30):
    print(x, end="&") 
print()
for roi_size in range(20, 112, 30):
    print(roi_size, end='&')
    for size in range(20, 112, 30):
        parts = data[str(roi_size)][str(size)]
        milliseconds = parts["Offset"]["mean"]
        seconds = parts["Offset"]["mean"] / 1000
        throughput_mb_s = batch_size_mb / seconds
        print("{:.1f}".format(throughput_mb_s), "&", sep='', end='')
    print("\\\\")
