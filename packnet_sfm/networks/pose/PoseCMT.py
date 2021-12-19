# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

#from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.depth.cmt2 import CMT_Ti
from packnet_sfm.networks.layers.resnet.pose_decoder import PoseDecoder

########################################################################################################################

class PoseCMT(nn.Module):
    """
    Pose network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, **kwargs):
        super().__init__()
        assert version is not None, "PoseResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.embed_dim = 46
        self.stem_channel = 16
        self.encoder = CMT_Ti(in_channels = 6, input_size = 256, embed_dim= 46, stem_channels = self.stem_channel)
        self.de_channels=[self.stem_channel, self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        self.decoder = PoseDecoder(self.de_channels, num_input_features=1, num_frames_to_predict_for=2)

    #def forward(self, target_image, ref_img):
    def forward(self, target_image, ref_imgs):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs = []

        # for i, ref_img in enumerate(ref_imgs):
        #     inputs = torch.cat([target_image, ref_img], 1)
        #     axisangle, translation = self.decoder([self.encoder(inputs)])
        #     outputs.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
        
        for i, ref_img in enumerate(ref_imgs):
            inputs = torch.cat([target_image, ref_img], 1)
            axisangle, translation = self.decoder([self.encoder(inputs)])
            outputs.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))


        pose = torch.cat(outputs, 1)
        return pose

########################################################################################################################

