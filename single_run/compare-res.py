import numpy as np
import math
def load_result_txt(fname):
    pos = []
    fnames = []
    data = []
    import time
    started = time.time()
    with open(fname) as fh:
        while True:
            print(".", end="", flush=True)
            line = fh.readline()
            if not line:
                break
            x, y, n, fn = line.rstrip().split(maxsplit=3)
            x, y, n = float(x), float(y), int(n)
            pos.append((x,y))
            fnames.append(fn)
            data.append(np.loadtxt(fh, max_rows=n))
    pos = np.asarray(pos)
    data = np.asarray(data)
    print("loaded in", time.time()-started, "s")
    return pos, fnames, data
    

pos, _, emida = load_result_txt("out-emida.txt")
pos, _, jove = load_result_txt("out-jove.txt")

#fs = 10  # fontsize
#pos = [1, 2, 4, 5, 7, 8]
#data = [np.random.normal(0, std, size=100) for std in pos]

#print(data)
counter = 0
data = []
for i in range(len(emida)):
    for j in range(len(emida[i])):
        for x in range(2, 4):
            val = abs(emida[i][j][x] - jove[i][j][x])
            
            if not math.isnan(val):
                if abs(jove[i][j][x]) > 0.0000001:
                    val = val / abs(jove[i][j][x])
                    data.append(val)
                    if(val > 0.1):
                        print(emida[i][j][x], jove[i][j][x])
                else:
                    #print(jove[i][j][x])
                    counter += 1
        


print(counter)
import numpy as np
import matplotlib.pyplot as plt

data = [data]
fig, axes = plt.subplots(figsize=(6, 6))

print(np.mean(data[0]))
print(np.percentile(data[0], 95))
print(np.percentile(data[0], 99))
print(np.percentile(data[0], 99.9))
print(np.percentile(data[0], 99.99))
print(np.max(data[0]))
print(0.1*len(data[0]))
axes.set_yscale('log')
axes.violinplot(data, [1], points=2000, widths=0.3,
                      showmeans=True, showextrema=True, showmedians=True)



plt.show()

#.0000518