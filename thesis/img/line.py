import json
import sys
import os
import math
import statistics
import numpy as np
import pprint
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import correlate

def arrow(y, beg, mid, end):
    val = 0
    step = 50/(mid-beg)
    for i in range(beg, mid):
        y[i] = val
        val += step
    step = 50/(end-mid)
    for i in range(mid, end):
        y[i] = val
        val -= step
    
    return y

def cube(y, beg, end):
    val = 0
    step = 0.5/70
    for i in range(beg, mid):
        y[i] = val
        val += step
    for i in range(mid, end):
        y[i] = val
        val -= step
        
    
    return y

if __name__ == '__main__':
    
    
    

    fig, ax = plt.subplots(figsize=(20,3))
    
    ax.set_ylabel("Statements")
    ax.set_xlabel('Programs')
    ax.set_title("Blah")
    #ax.set_axis_off()
    
    y = [np.sin(x) for x in np.arange(0,30,0.001)]
    y1 = [np.sin(x) for x in np.arange(0,3.14,0.001)]
    
    #y = arrow([0]*1000, 50,200,350)
    #y = arrow(y, 650, 700, 950)
    #y1 = arrow([0]*1000, 350,500,650)
    
    #y1 = cube(y1, 850, 950)
    
    #y = np.array(y)-np.mean(y)
    #y1 = np.array(y1)-np.mean(y1)
    
    cor = correlate(y, y1, mode='full')
    cor /= 100
    
    x = np.arange(len(y))
    
    ax.plot(x, y, label="Statements")
    ax.plot(np.arange(len(y1)) + 30000, y1, label="Statementss")
    ax.plot(np.arange(len(cor))+2000, cor, label="haha")
    
    
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())    
        
    #ax.legend()
   
    fig.tight_layout()

    plt.show()
    
    #fig.savefig("cross_corr.svg", dpi=fig.dpi)
