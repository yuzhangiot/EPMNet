# EPMNet
EPM-CONVOLUTION MULTILAYER-NETWORK FOR OPTICAL FLOW ESTIMATION

## Introduction
Deep learning has been tremendously successful in recent years leading to rapid progress in the process of optical flow estimation. However, these works overemphasize the factor of deep learning and ignore advantages of many traditional methods in optical flow estimation area. It leads to either low accuracy results or high redundancy models. In this paper, we combine residual flow based course-to-fine architecture and bilateral weights based patch match algorithm with deep learning, which turns out to achieve a competitive result. The high accuracy for both small and large motion estimation are mainly cause by two contributions: firstly, we present and implement an edge preserve patch match (EPM) layer that propagates self-similarity patterns in addition to offsets. The accuracy of optical flow prediction has greatly improved by this method. Secondly, we develop a course-to-fine network architecture to tackle large displacement estimation and introduce a residual flow method to solve small displacement estimation. 


## Quick Guide
First compile caffe, by configuring a

    "Makefile.config" (example given in Makefile.config.example)

then make with

    $ make -j 5 all tools pycaffe
    
    
## Running    
    Running a FlowNet on a single image pair ($net is a folder in models):

    $ run-flownet.py /path/to/$net/$net_weights.caffemodel[.h5] \
                     /path/to/$net/$net_deploy.prototxt.template \
                     x.png y.png z.flo
                     
    All script can be found in script folder.
