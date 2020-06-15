# coding: utf-8
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
data = [
0,
5,
0,
4,
1,
0,
2,
7,
4,
1,
0,
0,
0,
0,
2,
0,
0,
5,
3,
1,
0,
1,
1,
2,
0,
0,
0,
2,
0,
1]
sor = sorted(data)
sor
counts = [sor.count(x) for x in range(0,7)]
counts
counts = [sor.count(x) for x in range(0,8)]
counts
cumulative_counts = np.cumsum(counts)
plt.plot(range(0,8), cumulative_counts)
plt.show()
plt.hist(sor, bins=[0,1.1,3.1,7.1])
plt.show()
