# coding: utf-8
x=[17.79,
14.52,
17.21,
14.46,
16.47,
13.22,
16.27,
11.75,
15.53,
16.31,
14.92,
15.81,
14.72,
14.98,
14.63,
14.30,
14.23,
14.05,
13.70,
13.66]
sor = sorted(x)
len(sor)
sor[len(sor)*0.16]
len(sor)*0.16
import math
sor[math.ceil(len(sor)*0.16)-1]
sor
sixteen_perc = sor[math.ceil(len(sor)*0.16)-1]
eightyfour_perc = sor[math.ceil(len(sor)*0.84-1)]
len(sor)*0.84
eightyfour_perc
median = 0.5*(sor[len(sor)*0.5-1]+sor[len(sor)*0.5])
median = 0.5*(sor[int(len(sor)*0.5)-1]+sor[int(len(sor)*0.5)])
median
sor
math.median(sor)
import statistics
statistics.median(sor)
median
def percentile(perc, ls):
    idx = perc*len(ls)
    if idx-int(idx)!= 0: return ls[math.ceil(perc)-1]
    return 0.5*(ls[int(idx)-1]+ls[int(idx)])
    
percentile(25, sor)
percentile(0.25, sor)
percentile(0.75, sor)
import matplotlib.pyplot as plt
plt.boxplot(sor)
plt.show()
import numpy as np
seven = np.ones(7)
seven[0] = 11
plt.hist(sor, bins=np.cumsum(seven))
plt.show()

