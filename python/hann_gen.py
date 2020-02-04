#!/usr/bin/python3
import sys
import getopt
import random
from numpy import *

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
A = [[random.uniform(0, 256) for i in range(N)] for j in range(N)]
write_file(file + "_A.txt", A)

window = hanning(N)

A *= window[:,newaxis]
A *= window[newaxis,:]

write_file(file + "_res.txt", A)

print("Done")