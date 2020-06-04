# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:34:24 2020

@author: win10
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:00:17 2020

@author: win10
"""

import torch
import os

from networks import CUNet
from torch.utils.data import DataLoader 
from utils import  MMSRDataset, mkdir
from kornia.losses import psnr_loss, ssim
from kornia.color import rgb_to_grayscale
from scipy.io import savemat

test_path = 'test_results'
mkdir(test_path)
batch_size = 32
lr = 1e-4
num_epoch = 200
scale = 2

n_layer = 4
ks = 7
n_feat=64
n_channel=1
net = CUNet(ks=ks, 
                 n_feat=n_feat, 
                 n_layer=n_layer, 
                 n_channel=n_channel).cuda()

net.load_state_dict(torch.load(os.path.join(r'MMSR_logs\06-02-15-31\last_net.pth'))['net'])
testloader = DataLoader(MMSRDataset(r'D:\data\MMSR\scale2\test', scale),      
                              batch_size=1)

metrics = torch.zeros(2,testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (lr, guide, gt) in enumerate(testloader):
        lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
        guide = rgb_to_grayscale(guide)
        lr = torch.nn.functional.interpolate(lr, scale_factor=scale, mode='bicubic',align_corners=True)
        imgf = [net(lr[:,k,:,:].unsqueeze(1), guide) for k in range(lr.shape[1])]
        imgf = torch.clamp(torch.cat(imgf, dim=1), min=0., max=1.)
        metrics[0,i] = psnr_loss(imgf, gt, 1.)
        metrics[1,i] = 1-2*ssim(imgf, gt, 5, 'mean', 1.)
        savemat(os.path.join(test_path,testloader.dataset.files[i].split('\\')[-1]),
               {'HR':imgf.squeeze().detach().cpu().numpy()} )

import xlwt
f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
img_name = [i.split('\\')[-1].replace('.mat','') for i in testloader.dataset.files]
metric_name = ['PSNR','SSIM']
for i in range(len(metric_name)):
    sheet1.write(i+1,0,metric_name[i])
for j in range(len(img_name)):
   sheet1.write(0,j+1,img_name[j])  # 顺序为x行x列写入第x个元素
for i in range(len(metric_name)):
    for j in range(len(img_name)):
        sheet1.write(i+1,j+1,float(metrics[i,j]))
sheet1.write(0,len(img_name)+1,'Mean')
for i in range(len(metric_name)):
    sheet1.write(i+1,len(img_name)+1,float(metrics.mean(1)[i]))
f.save(os.path.join(test_path,'test_result.xls'))
