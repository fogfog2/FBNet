# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from functools import partial


from packnet_sfm.networks.depth.cmt2 import CMT, CMT_Ti, CMT_B

from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth
import torch.nn.functional as F


class DepthCMT2(nn.Module):
    def __init__(self, version=None, 
                 embed_dim=46):
        super().__init__()


        
        self.embed_dim= embed_dim    
        self.channels = 256
        self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        self.de_channels=[16, self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      

        self.cmt = CMT_Ti(in_channels = 3, input_size = 256, embed_dim= 46)
        #self.cmt = CMT_B(in_channels = 3, input_size = 256, embed_dim= 76)

        self.decoder = DepthDecoder(num_ch_enc=self.de_channels)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x):

        #swin encoder
        x = self.cmt(x)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

       

