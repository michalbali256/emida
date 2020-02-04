#!/usr/bin/python3
import sys
import getopt
import random

def write_file(filename, matrix):
  f = open(filename, "w")
  f.write(str(len(matrix)) + "\n")
  a = "\n".join([" ".join(map(str, j)) for j in matrix])
  f.writelines(a)
  f.close()


file = ''
help = 'mat_gen.py -f <fileName> -N <matrixSize>'
N = 32
try:
  opts, args = getopt.getopt(sys.argv[1:],"hN:f:", ["--file="])
except getopt.GetoptError:
  print(help)
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
    print(help)
    sys.exit()
  elif opt in ("-f", "--file"):
    file = arg
  elif opt in ("-N"):
    N = int(arg)



print("Generating")
MAXVAL = 256
A = [[random.randint(0, 256) for i in range(N)] for j in range(N)]
#write_file(file + "_A.txt", A)
B = [[random.randint(0, 256) for i in range(N)] for j in range(N)]
#write_file(file + "_B.txt", B)

from scipy.signal import correlate
from time import time
print("Calculating")
start = time()
cor = correlate(A, B, mode='full')
end = time()
print("End")
print(end - start)

#write_file(file + "_res.txt", cor)
print("Max:" + str(max(map(max, cor))))