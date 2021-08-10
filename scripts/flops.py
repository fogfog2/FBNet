import torch
import torchvision.models as models
from packnet_sfm.networks.depth import DepthResNet

from packnet_sfm.networks.depth.DepthCMT2 import DepthCMT2
from packnet_sfm.networks.depth.cmt2 import CMT, CMT_Ti, CMT_B

from ptflops import get_model_complexity_info

with torch.cuda.device(0):
#  net = models.resnet18()

  #net = DepthResNet()
  net = CMT_Ti(embed_dim= 46)
  macs, params = get_model_complexity_info(net, (3, 640, 192), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))