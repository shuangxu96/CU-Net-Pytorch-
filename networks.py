# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:49:27 2020

@author: BSawa
"""

import torch 
import torch.nn as nn

class SST(nn.Module):
    def __init__(self, n_parameter=1):
        super(SST, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, n_parameter, 1, 1))
        nn.init.xavier_uniform_(self.gamma)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        gamma = torch.clamp(self.gamma, min=0.)
        return torch.sign(x)*torch.clamp(x.abs()-gamma, min=0.)
    
class CUNet(nn.Module):
    def __init__(self, ks=7, 
                 n_feat=64, 
                 n_layer=4, 
                 n_channel=3):
        super(CUNet, self).__init__()
        padding = int((ks-1)/2)
        self.n_layer = n_layer
        
        # get u
        self.decoder_u0 = nn.Conv2d(n_channel, n_feat, ks, 1, padding)
        self.sst_u0 = SST(n_feat)
        self.decoder_u = nn.ModuleList([nn.Conv2d(n_feat, n_channel, ks, 1, padding) for i in range(n_layer)])
        self.encoder_u = nn.ModuleList([nn.Conv2d(n_channel, n_feat, ks, 1, padding) for i in range(n_layer)])
        self.sst_u = nn.ModuleList([SST(n_feat) for i in range(n_layer)])
        
        # get v
        self.decoder_v0 = nn.Conv2d(n_channel, n_feat, ks, 1, padding)
        self.sst_v0 = SST(n_feat)
        self.decoder_v = nn.ModuleList([nn.Conv2d(n_feat, n_channel, ks, 1, padding) for i in range(n_layer)])
        self.encoder_v = nn.ModuleList([nn.Conv2d(n_channel, n_feat, ks, 1, padding) for i in range(n_layer)])
        self.sst_v = nn.ModuleList([SST(n_feat) for i in range(n_layer)])
        
        # get z
        self.recon_x = nn.Conv2d(n_feat, n_channel, ks, 1, padding)
        self.recon_y = nn.Conv2d(n_feat, n_channel, ks, 1, padding)
        self.decoder_c0 = nn.Conv2d(2*n_channel, n_feat, ks, 1, padding)
        self.sst_c0 = SST(n_feat)
        self.decoder_c = nn.ModuleList([nn.Conv2d(n_feat, 2*n_channel, ks, 1, padding) for i in range(n_layer)])
        self.encoder_c = nn.ModuleList([nn.Conv2d(2*n_channel, n_feat, ks, 1, padding) for i in range(n_layer)])
        self.sst_c = nn.ModuleList([SST(n_feat) for i in range(n_layer)])
        
        # reconstructor
        self.rec_u = nn.Conv2d(n_feat, n_channel, ks, 1, padding)
        self.rec_v = nn.Conv2d(n_feat, n_channel, ks, 1, padding)
        self.rec_c = nn.Conv2d(n_feat, n_channel, ks, 1, padding)
        
    def get_u(self, x):
        # get the code of image x
        p1 = self.decoder_u0(x) # [B,n_feat,H,W]
        tensor = self.sst_u0(p1)
        for i in range(self.n_layer):
            p3 = self.decoder_u[i](tensor) # [B,n_channel,H,W]
            p4 = self.encoder_u[i](p3)
            p5 = tensor-p4
            p6 = p1+p5
            tensor = self.sst_u[i](p6)
        return tensor
    
    def get_v(self, y):
        # get the code of image y
        p1 = self.decoder_v0(y) # [B,n_feat,H,W]
        tensor = self.sst_v0(p1)
        for i in range(self.n_layer):
            p3 = self.decoder_v[i](tensor) # [B,n_channel,H,W]
            p4 = self.encoder_v[i](p3)
            p5 = tensor-p4
            p6 = p1+p5
            tensor = self.sst_v[i](p6)
        return tensor
    
    def get_c(self, z):
        p1 = self.decoder_c0(z) # [B,n_feat,H,W]
        tensor = self.sst_c0(p1)
        for i in range(self.n_layer):
            p3 = self.decoder_c[i](tensor) # [B,2*n_channel,H,W]
            p4 = self.encoder_c[i](p3)
            p5 = tensor-p4
            p6 = p1+p5
            tensor = self.sst_c[i](p6)
        return tensor

    
    def forward(self, x, y):
        u = self.get_u(x) # [B,n_feat,H,W]
        v = self.get_v(y) # [B,n_feat,H,W]
        p8x = x-self.recon_x(u)
        p8y = y-self.recon_y(v)
        p9xy = torch.cat((p8x,p8y), dim=1)
        c = self.get_c(p9xy)
        
        f = self.rec_c(c)+self.rec_u(u)+self.rec_v(v)
        return f

