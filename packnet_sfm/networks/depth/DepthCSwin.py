# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from functools import partial


from packnet_sfm.networks.depth.cswin import CSWinTransformer

import torch.nn.functional as F
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth

########################################################################################################################

class DepthCSwin(nn.Module):
    def __init__(self, version=None, 
                 embed_dim=96):
        super().__init__()


        
        self.embed_dim= embed_dim    
        self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]         
        init_dim = self.embed_dim/2
        self.de_channels=[init_dim, self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      

        #
        self.split_size_w = [1,2,int(640/32),int(640/32)]
        self.split_size_h = [1,2,int(192/32),int(192/32)]
        self.cswin = CSWinTransformer(img_width =640, img_height =192, patch_size=4, embed_dim=64, depth=[1,2,21,1],
        split_size_w=self.split_size_w, split_size_h=self.split_size_h,num_heads=[2,4,8,16], mlp_ratio=4.)


        self.decoder = DepthDecoder(num_ch_enc=self.de_channels)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x):

        #swin encoder
        x = self.cswin(x)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

       


########################################################################################################################
def resize(input, size=None, scale_factor=None,mode='nearest',align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=4, mode="nearest")


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth