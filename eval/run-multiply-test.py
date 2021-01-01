#!/usr/bin/python3

from math import sqrt
def hex_pos(size, step):
    m = int(size/step/sqrt(0.75))
    n = int(size/step)
    for j in range(m+1):
        for i in range(n-j%2+1):
            x = (i+j%2/2)*step
            y = j*sqrt(0.75)*step
            yield x, y

import numpy as np
import sys
import pprint as pp
import statistics as stat
import time
import subprocess
import json
#pp = pprint.pprint()

def read_roi(fname):
    with open(fname) as fh:
        s = np.loadtxt(fh, dtype=int, max_rows=1)
        pos = np.loadtxt(fh,  dtype=int, ndmin=2)
    return s, pos


def get_times(args):
    
    exec_name = "../build/x64-Release/bin/multiply_test"
    final_args = [exec_name]
    final_args.extend(args)
    
    started = time.time()
    cmd = subprocess.Popen(final_args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    TOTAL_TIME = (time.time()-started)*1000.0
    
    output = cmd_err.decode(sys.stdout.encoding)
    #print(output)
    times = {}
    times["TOTAL_TIME"] = TOTAL_TIME
    for line in output.split('\n'):
        if line.strip() == '':
            continue
        time_s = line.split(' ')[-1]
        label = line.split(':')[0]
        try:
            times[label] = float(time_s[0:-3])
        except ValueError:
            pass
    
    return times

def run_emida(args = [], repeat_times = 10):
    
    total_times = {}
    for i in range(repeat_times):
        nt = get_times(args)
        for k in nt.keys():
            if not k in total_times:
                total_times[k] = []
            total_times[k].append(nt[k])
        print('.', end='', flush=True)
        #pp.pprint(nt)
    stats = {}
    
    for k in nt.keys():
        vals = total_times[k]
        stats[k] = {}
        stats[k]['mean'] = stat.mean(vals)
        stats[k]['stdev'] = stat.stdev(vals)
        if stat.mean(vals) == 0:
            stats[k]['stdev%'] = 0
        else:
            stats[k]['stdev%'] = stat.stdev(vals) / stat.mean(vals)
        stats[k]['min'] = min(vals)
        stats[k]['max'] = max(vals)
    
    
    #pp.pprint(total_times)
    #pp.pprint(stats)
    return stats

def write_roi(fname, s, pos, n):
    
    with open(fname,"w") as fh:
        print(s, file=fh)
        for i in range(0,n):
            print(pos[i][0], pos[i][1], file=fh)

if __name__ == "__main__":
    args = ["10", "10", "5"]
    stats = run_emida(args, 2)
    plot = {}
    for batch_size in range(7, 8):
        plot[batch_size] = {}
        for roi_count in range(20, 120, 30):
            plot[batch_size][roi_count] = {}
            for size in range (20,113, 10):
                args = [str(size), str(roi_count), str(batch_size)]
                print("\n", size, end='', sep='')
                stats = run_emida(args, 8)
                plot[batch_size][roi_count][size] = stats
    
    with open("out-graph.json","w") as fh:
        json.dump(plot, fh, indent=4)