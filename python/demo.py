#! /usr/bin/python
# Author: Ruoteng Li
# Date: 6th Aug 2016
"""
Demo.py
This file demonstrates how to use kittitool module to read
 and visualize flow file saved in kitti format .png
"""
import matplotlib
matplotlib.use('Agg')
from lib import flowlib as fl
import sys


# read Middlebury format optical flow file (.flo)
print "Visualizing Middlebury flow example ..."
flow_file_Middlebury = sys.argv[1]
flow_Middlebury = fl.read_flow(flow_file_Middlebury)
fl.visualize_flow(flow_Middlebury)
