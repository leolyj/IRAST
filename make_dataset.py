import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import cv2



img_paths = []

with open('./UCF-QNRF_train_with_val.json', 'r') as outfile:
     img_paths = json.load(outfile)


img_target=[]
num_label=0
num_unlabel=0
for img_path in img_paths:
    print img_path
    index = img_paths.index(img_path)
    print index
    # using this code for parta & partb
    #mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')) 

    # using this code for ucf-qnrf
    mat = io.loadmat(img_path.replace('.jpg', '_ann.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_')) 

    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))

    # using this code for part a & part b
    #gt = mat["image_info"][0, 0][0, 0][0] 


    # using this code for ucf-qnrf
    gt = mat['location_target'] 

    gt = gt
    print gt.shape[0]
    gt_value = gt.shape[0]

    # for validation or testing set in all dataset 
    #if(index<0):

    #for part a training set
    #if ((index<=80)or(index>=93 and index<=172)or(index>=184 and index<=213)or(index>=218 and index<=234)or(index>=237 and index<=238)):

    #for part b training set
    #if ((index<=139)or(index>=160 and index<=251)or(index>=265 and index<=296)or(index>=301 and index<=312)or(index>=315 and index<=318)):

    #for ucf-qnrf training set
    if ((index<255)or(index>=341 and index<506)or(index>=560 and index<664)or(index>=698 and index<743)or(index>=757 and index<811)or(index>=829 and index<857)or(index>=867 and index<901) or(index>=913 and index<938) or (index>=946 and index<953) or (index>=956 and index<960)):
    
        print 'generate unlabeled image without density map supervision'
        num_unlabel = num_unlabel +1

        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
             hf['count_value'] = gt_value
             

    else:
        num_label = num_label +1
        print 'generate with map'
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1

        d = gaussian_filter(k,3,truncate=4)

        print d.sum()

        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:

             hf['density'] = d

print img_target
print len(img_target)
print 'num_label',num_label
print 'num_unlabel',num_unlabel





