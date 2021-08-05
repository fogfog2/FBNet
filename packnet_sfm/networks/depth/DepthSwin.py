# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from functools import partial


from packnet_sfm.networks.depth.SwinTransformer import SwinTransformer

import torch.nn.functional as F


########################################################################################################################

class lconv(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super().__init__()
        self.conv =  nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        #out = self.nonlin(out)
        ##norm?
        return out

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
class fpnconv(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super().__init__()
        self.conv =  nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.nonlin = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(self.bn1(out))
        ##norm?
        return out

class DepthSwin(nn.Module):
    def __init__(self, version=None, 
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 drop_path_rate=0.2,
                 ape=False):
        super().__init__()


        
        self.embed_dim= embed_dim    
        self.depths=depths
        self.num_heads =num_heads
        self.drop_path_rate = drop_path_rate
        self.ape = ape       
        self.channels = 128

        self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
   
        self.swin = SwinTransformer(embed_dim= self.embed_dim,
                                    depths=self.depths,
                                    num_heads=self.num_heads,
                                    drop_path_rate=self.drop_path_rate,
                                    ape = self.ape)
        self.lateral_convs = nn.ModuleList()
        #self.fpn_convs = nn.ModuleList()
        #self.fc_convs = nn.ModuleList()

        for i , in_channel in enumerate( self.in_channels):
            l_conv =  lconv(in_channel , self.channels)
            self.lateral_convs.append(l_conv)
            

        self.fpn_conv = fpnconv(self.channels , self.channels)
        self.fpn_conv2 = fpnconv(self.channels*2 , self.channels)
        self.fc_conv = fpnconv(self.channels , 1)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        #swin encoder
        x = self.swin(x)

        #build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        #build top-down path 
        used_backbone_levels = len(laterals)

        laterals[3] = self.fpn_conv(laterals[3])

        for i in range(used_backbone_levels-1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]

            #add 
            # laterals[i-1] += resize(
            #     laterals[i],
            #     size=prev_shape,
            #     mode='bilinear',
            #     align_corners=False)

            #concat
            prev = [laterals[i-1]]
            prev += [resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=False)]
            cc = torch.cat(prev, 1 )
            laterals[i-1]= self.fpn_conv2(cc)


        #fpn_outs = [self.fpn_conv(laterals[i]) for i in range(used_backbone_levels)]
        fc_outs = [self.fc_conv(laterals[i]) for i in range(used_backbone_levels)]
        outs = [self.scale_inv_depth(self.sigmoid(upsample(fc_outs[i])))[0] for i in range(used_backbone_levels)]
        

        # #upsampling add decoder
        # for i in range( len(x) ,0, -1):
        #     x[i] = self.conv

        # x[0]= upsample(self.conv1(x[0]))
        # x[1]= upsample(self.conv2(x[1]))
        # x[2]= upsample(self.conv3(x[2]))
        # x[3]= upsample(self.conv4(x[3]))

        #direct upsampling
        # x[0]= self.scale_inv_depth(self.sigmoid(upsample(self.squeeze(self.conv1(x[0])))))[0]
        # x[1]= self.scale_inv_depth(self.sigmoid(upsample(self.squeeze(self.conv2(x[1])))))[0]
        # x[2]= self.scale_inv_depth(self.sigmoid(upsample(self.squeeze(self.conv3(x[2])))))[0]
        # x[3]= self.scale_inv_depth(self.sigmoid(upsample(self.squeeze(self.conv4(x[3])))))[0]
        # x[0]= self.scale_inv_depth(self.sigmoid(upsample(self.conv1(x[0]))))[0]
        # x[1]= self.scale_inv_depth(self.sigmoid(upsample(self.conv2(x[1]))))[0]
        # x[2]= self.scale_inv_depth(self.sigmoid(upsample(self.conv3(x[2]))))[0]
        # x[3]= self.scale_inv_depth(self.sigmoid(upsample(self.conv4(x[3]))))[0]


        return outs

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