import h5py
import torch
import shutil
import random
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math


            
def save_checkpoint0(state, is_best,task_id, filename='checkpoint.tar'):
    torch.save(state, task_id+filename)
    if is_best:
         shutil.copyfile(task_id+filename, task_id+'model_best.tar')

def densitymap_to_densitylevel(density_map,th1,th2):
    mask_0 = (density_map <= 0).long()

    mask_1 = ((density_map > 0)&(density_map <= th1)).long()

    mask_2 = ((density_map > th1)&(density_map <= th2)).long()

    mask_3 = (density_map > th2).long()

    level = 0 * mask_0 + 1 * mask_1 + 2*mask_2 +3*mask_3

    cls_pesudo_label = level.squeeze(0)
    return cls_pesudo_label

def densitymap_to_densitymask(density_map,threshold):
    density_mask = (density_map > threshold).long()
    density_mask = density_mask.squeeze(0)
    return density_mask


def unlabel_CE_loss2v1(logits2,prob3,prob4,max_update_pixel,th,criterion_cls):
    prob2 = torch.nn.functional.softmax(logits2,dim=1)
    prob_max = torch.max(prob2,dim=1)[0]
    target_temp = torch.argmax(prob2,dim=1)
  
    # choose the valid pixels
    mask = ((prob2[:, 1, :, :] > th) | 
    ((prob2[:, 0, :, :] > th) & (prob3[:, 0, :, :] > th) & (prob4[:, 0, :, :] > th))).float()
  
    target = ((mask * target_temp.float()) + (10*(1-mask))).long()
    loss_ce_ori = criterion_cls(logits2,target)
    return loss_ce_ori




def unlabel_CE_loss3v1(logits3, prob2, prob4, max_update_pixel, th, criterion_cls):
    prob3 = torch.nn.functional.softmax(logits3, dim=1)
    prob_max = torch.max(prob3, dim=1)[0]
    target_temp = torch.argmax(prob3, dim=1)

     # choose the valid pixels
    mask = (((prob2[:, 1, :, :] > th) & (prob3[:, 1, :, :] > th)) | 
    ((prob3[:, 0, :, :] > th) & (prob4[:, 0, :, :] > th))).float()
   
    target = ((mask * target_temp.float()) + (10 * (1 - mask))).long()
    loss_ce_ori = criterion_cls(logits3, target)
    return loss_ce_ori



def unlabel_CE_loss4v1(logits4, prob2, prob3,  max_update_pixel, th, criterion_cls):
    prob4 = torch.nn.functional.softmax(logits4, dim=1)
    prob_max = torch.max(prob4, dim=1)[0]
    target_temp = torch.argmax(prob4, dim=1)

     # choose the valid pixels
    mask = (((prob2[:, 1, :, :] > th)& (prob3[:, 1, :, :] > th) & (prob4[:, 1, :, :] > th)) 
    | ((prob4[:, 0, :, :] > th))).float()
  
    target = ((mask * target_temp.float()) + (10 * (1 - mask))).long()
    loss_ce_ori = criterion_cls(logits4, target)
    return loss_ce_ori
