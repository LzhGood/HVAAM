#coding=UTF-8

import h5py

f = h5py.File('weights.sam-vgg.00-1.5911.pkl','r')   #打开h5文件
f=["model_weights"]
print (f.keys())