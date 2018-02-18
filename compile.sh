#!/bin/bash
sudo apt-get install libprotobuf-dev libhdf5-serial-dev libatlas-base-dev libsnappy-dev libleveldb-dev liblmdb-dev protobuf-compiler
sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so /usr/local/lib/libhdf5_hl.so
sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/local/lib/libhdf5.so
git clone ssh://git@loft.party:2202/ML/cudnn.git
sudo rm /usr/local/cuda/lib64/libcudnn*
sudo cp cudnn/include/* /usr/local/cuda/include/
sudo cp cudnn/lib64/* /usr/local/cuda/lib64/
sudo ldconfig

make clean
make -j 5 all tools pycaffe distribute
