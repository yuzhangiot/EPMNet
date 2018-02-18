from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys, numpy as np
from subprocess import call
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#get lines like "I0803 05:33:07.420920 16208 solver.cpp:406]   Test net output #0: flow_loss2 = 4915.2 (* 0.005 = 24.576 loss)"
try:
	if sys.argv[3]=='test':
		grepstr = "Test net output #" + sys.argv[2]
	else:
		grepstr = "Train net output #" + sys.argv[2]
except:
	grepstr = "Train net output #" + sys.argv[2]
pcmd = "grep -a \'{}\' ".format(grepstr) + sys.argv[1]
sys.stdout.write(str(pcmd)+'\n')
raw_input()

out = os.popen(pcmd)
# print(out.read())

results = []
#read loss in these lines

for line in out:
	try:
		edi = line.index('(') - 1
	except:
		edi = len(line)
	try:
		results.append(float(line[line.index('=') + 2 : edi]))
	except:
		sys.stdout.write(str(line)+'\n') 
		sys.stdout.write(str(line.index('=') + 2)+'\n')
		sys.stdout.write(str(edi)+'\n')
		raise

# print(len(results))
# print(results)

y1 = results[1:]
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

sys.stdout.write("ploting..."+'\n')
# out_png = '/home/joseph/workbench/flownet2_fix/models/spycorr/logout.png'
out_png = sys.argv[1] + '-' + sys.argv[2]
try:
	if sys.argv[3]  == 'test':
		out_png += '-test'
except:
	pass
out_png += '.png'
plt.savefig(out_png, dpi=600)
# plt.show()
