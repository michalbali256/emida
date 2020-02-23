#!/usr/bin/python3
import sys
import getopt
import random

print('gen')

from generator import *

def write_file(filename, matrix):
  f = open(filename, "w")
  f.write(str(len(matrix)) + "\n")
  a = "\n".join([" ".join(map(str, j)) for j in matrix])
  f.writelines(a)
  f.close()


file = ''
help = 'mat_gen.py -f <fileName>'
try:
  opts, args = getopt.getopt(sys.argv[1:],"hf:", ["--file="])
except getopt.GetoptError:
  print(help)
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
    print(help)
    sys.exit()
  elif opt in ("-f", "--file"):
    file = arg

ref, tran = generate()

write_file(file + "_temp.txt", ref)

write_file(file + "_pic.txt", tran)
