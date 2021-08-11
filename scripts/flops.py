import torch
import torchvision.models as models
from packnet_sfm.networks.depth.DepthResNet import DepthResNet

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder

from packnet_sfm.networks.depth.DepthCMT2 import DepthCMT2

from packnet_sfm.networks.depth.DepthCSwin import DepthCSwin

from packnet_sfm.networks.depth.cmt2 import CMT, CMT_Ti, CMT_B

from packnet_sfm.networks.depth.cswin import CSWinTransformer

from packnet_sfm.networks.depth.SwinTransformer2 import SwinTransformer

from packnet_sfm.networks.depth.DepthSwin2 import DepthSwin2

from ptflops import get_model_complexity_info

with torch.cuda.device(0):
#  net = models.resnet18()

  #net = DepthResNet(version="18pt")
  
  #net = ResnetEncoder(num_layers=18, pretrained="pt")

  #net = CMT_Ti(embed_dim= 46)

  #net = DepthCSwin(embed_dim= 64)

#   split_size_w = [1,2,int(640/32),int(640/32)]
#   split_size_h = [1,2,int(192/32),int(192/32)]
#   net = CSWinTransformer(img_width =640, img_height =192, patch_size=4, embed_dim=64, depth=[1,2,21,1],
#         split_size_w=split_size_w, split_size_h=split_size_h,num_heads=[2,4,8,16], mlp_ratio=4.)

  #net = DepthSwin2()

  net = SwinTransformer(embed_dim= 96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    drop_path_rate=0.2,
                                    ape = False,
                                    init_dim = 48)

  macs, params = get_model_complexity_info(net, (3, 640, 192), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))