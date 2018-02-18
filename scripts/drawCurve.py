from __future__ import print_function

import os, sys, numpy as np
from subprocess import call
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#get lines like "I0803 05:33:07.420920 16208 solver.cpp:406]   Test net output #0: flow_loss2 = 4915.2 (* 0.005 = 24.576 loss)"
out = os.popen("grep 'Test net output #0' " + sys.argv[1])
# print(out.read())

results = []
#read loss in these lines
for line in out:
	results.append(float(line[line.index('=') + 2 : line.index('(') - 1]))
	# print(line[line.index('=') + 2 : line.index('(') - 1])

# print(len(results))
# print(results)

# x1 = (0, 50, 100, 200, 400, 600, 1000, 2000, 3000, 4000, 5000)

# y1 = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
# y1 = (3.84347, 3.84347, 3.98493, 4.25461, 4.71211, 5.31234, 6.95694, 9.47803, 12.6878, 15.8233,17.4797)
y1 = results
x1 = range(0, len(y1))

# print str(len(x1)) + ";;;" + str(len(y1))

plt.subplot(1, 1, 1)
plt.plot(x1, y1, 'y.-')
# plt.title('A tale of 2 subplots')
# plt.ylabel('Damped oscillation')

# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, 'r.-')
plt.xlabel('iteration numbers')
plt.ylabel('loss')

plt.show()