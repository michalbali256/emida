N = 415
x = [[N*j + i for i in range(N)] for j in range(N)]

from scipy.signal import correlate
from time import time
start = time();
cor = correlate(x, x, mode='full')
end = time();
print(end - start)

#print(cor)


