# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:00:17 2020

@author: win10
"""

import torch
import datetime
import os

from networks import CUNet
from torch.utils.data import DataLoader 
from tensorboardX import SummaryWriter
import torch.nn as nn
from utils import H5Dataset, prepare_data, MMSRDataset
from kornia.losses import psnr_loss, ssim
from kornia.color import rgb_to_grayscale

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
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                 milestones=[50,100,150], 
                                                 gamma=0.9)

# loaders
prepare_data_flag = False # set it to False, if you have prepared dataset
if prepare_data_flag is True:
    prepare_data(data_path = r'D:\data\MMSR\scale2', 
                     patch_size=32, aug_times=4, stride=25, synthetic=True, scale=2,
                     file_name=r'cave_train.h5'
                     )
trainloader      = DataLoader(H5Dataset(r'cave_train.h5'),      
                              batch_size=batch_size, shuffle=True) #[N,C,K,H,W]
validationloader = DataLoader(MMSRDataset(r'D:\data\MMSR\scale2\validation',scale),      
                              batch_size=1)
loader = {'train':      trainloader,
          'validation': validationloader}
    
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join('MMSR_logs',timestamp)
writer = SummaryWriter(save_path)


# training
step = 0
best_psnr_val = 0.
torch.backends.cudnn.benchmark = True
for epoch in range(num_epoch):
    ''' train '''
    for i, (lr, guide, gt) in enumerate(loader['train']):
        lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
        tmp_ind = int(torch.randint(lr.shape[1],[1]))
        lr = torch.nn.functional.interpolate(lr[:,tmp_ind,:,:].unsqueeze(1), scale_factor=scale, mode='bicubic',align_corners=True)
        gt = gt[:,tmp_ind,:,:].unsqueeze(1)
        guide = rgb_to_grayscale(guide)
        #1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        imgf = net(lr, guide)
        loss = nn.MSELoss()(gt, imgf)
        loss.backward()
        optimizer.step()
        
        #2.  print
        print("[%d,%d] Loss: %.4f" %
              (epoch+1, i+1, loss.item()))
        #3. Log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        step+=1
        
    ''' validation ''' 
    psnr_val = 0.
    with torch.no_grad():
        net.eval()
        for j, (lr, guide, gt) in enumerate(loader['validation']):
            lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
            guide = rgb_to_grayscale(guide)
            lr = torch.nn.functional.interpolate(lr, scale_factor=scale, mode='bicubic',align_corners=True)
            imgf = [net(lr[:,k,:,:].unsqueeze(1), guide) for k in range(lr.shape[1])]
            imgf = torch.clamp(torch.cat(imgf, dim=1), min=0., max=1.)

            psnr_val += psnr_loss(imgf, gt, 1.)
        psnr_val = float(psnr_val/loader['validation'].__len__())
    writer.add_scalar('PSNR on validation data', psnr_val, epoch)

    
    ''' save model ''' 
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'epoch':epoch},
                os.path.join(save_path, 'last_net.pth'))
    scheduler.step()
    

'''
Test
'''
from scipy.io import savemat
net.load_state_dict(torch.load(os.path.join(save_path,'last_net.pth'))['net'])
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
        savemat(os.path.join(save_path,testloader.dataset.files[i].split('\\')[-1]),
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
f.save(os.path.join(save_path,'test_result.xls'))
