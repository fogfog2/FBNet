# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from functools import partial

import numpy as np

from packnet_sfm.networks.depth.cmt_h import CMT_Layer

from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth
import torch.nn.functional as F

import torchvision.models as models

class fcconv(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super().__init__()
        self.conv =  nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.nonlin = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(self.bn1(out))
        ##norm?
        return out
    
class DepthCMT_H(nn.Module):
    def __init__(self, version=None, 
                 num_layers = 18, pretrained = True, input_height = 192, input_width = 640, start_layer = 2, use_upconv = True, embed_dim=46):
        super().__init__()


        
        self.embed_dim= embed_dim    
        self.channels = 256
        self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        self.de_channels=[64, self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        
        
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        for i in range(start_layer,5):
            value = 2**(i-1)
            self.num_ch_enc[i] = self.embed_dim * value
            
            
        self.cmt_start_layer = start_layer
        self.use_upconv = use_upconv
        #self.cmt = CMT_Ti(in_channels = 3, input_size = 256, embed_dim= 46)
        self.cmt = CMT_Layer(input_width = input_width, input_height= input_height, embed_dim= self.embed_dim, start_layer=self.cmt_start_layer)
        #self.cmt = CMT_B(in_channels = 3, input_size = 256, embed_dim= 76)
        
        self.res_ch_enc = np.array([64, 64, 128, 256, 512])

        self.upconv = fcconv(self.res_ch_enc[1],self.embed_dim)
        self.upconv2 = fcconv(self.res_ch_enc[2],self.embed_dim*2)
        self.upconv3 = fcconv(self.res_ch_enc[3],self.embed_dim*4)
        
     
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}


        num_layers = 18
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        encoder = resnets[num_layers](pretrained)
        
        self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        
        self.decoder = DepthDecoder(num_ch_enc=self.num_ch_enc)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x):

        #encoder
        
        self.features = []
        x = (x - 0.45) / 0.225
        x= self.layer0(x)       
        self.features.append(x) 
        x= self.layer1(x)
        self.features.append(x)
        
        
        if self.use_upconv:
            out = self.upconv(x)
        else:
            out = x
        
        if self.cmt_start_layer>2:
            out = self.layer2(out)                     
            self.features.append(out)
            if self.use_upconv:
                out = self.upconv2(out)
            else:
                out = out       
                
            if self.cmt_start_layer>3:             
                out = self.layer3(out)
                self.features.append(out)
                if self.use_upconv:
                    out = self.upconv3(out)
                else:
                    out = out             
                    
                    
        cmt_out = self.cmt(out)
        self.features = self.features + cmt_out
        
        
        x = self.decoder(self.features)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

       

