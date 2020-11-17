


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
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
from utils import densitymap_to_densitymask,densitymap_to_densitylevel





from torchvision import datasets, transforms

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
#
with open('./part_A_test.json', 'r') as outfile:
    img_paths = json.load(outfile)

model = CSRNet()
model = model.cuda()
model.eval()

checkpoint = torch.load('0model_best_IRAST_parta.tar')
print checkpoint['epoch']

model.load_state_dict(checkpoint['state_dict'])


mae = 0
mse=0

for i in xrange(len(img_paths)):
    print img_paths[i]

    img = Image.open(img_paths[i])
    img = transform(img.convert('RGB')).cuda()

    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth_d = np.asarray(gt_file['density'])

    groundtruth_d = cv2.resize(groundtruth_d, (groundtruth_d.shape[1] / 8, groundtruth_d.shape[0] / 8),
                               interpolation=cv2.INTER_CUBIC) * 64


    d1,_,_,_ = model(img.unsqueeze(0),0)

    print 'et:',(d1).detach().cpu().sum().numpy()
    print 'gt:',np.sum(groundtruth_d)

    mae += abs((d1).detach().cpu().sum().numpy()  - np.sum(groundtruth_d))
    mse += ((d1).detach().cpu().sum().numpy() - np.sum(groundtruth_d))**2

    print i,mae



print 'mae',mae/len(img_paths)
print 'mse',np.sqrt(mse/len(img_paths))



