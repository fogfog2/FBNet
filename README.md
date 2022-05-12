This ropository is referenced from https://github.com/TRI-ML/packnet-sfm

add transformer backbone for monoculdar depth estimation

            swin 
            https://arxiv.org/abs/2103.14030
            https://github.com/microsoft/Swin-Transformer

            cswin
            https://arxiv.org/abs/2107.00652
            https://github.com/microsoft/CSWin-Transformer

            cmt
            https://arxiv.org/abs/2107.06263
            https://github.com/yuranusduke/CMT-Convolutional-NN-Meets-ViT


add fbnet 
 - [fbnet]

vs code args


            "cwd": "${workspaceFolder}",
            "env": {"PYTHONPATH": "${cwd}" }
            
            "program": "${workspaceFolder}/scripts/train_origin.py",
            "args": [  "configs/overfit_kitti_swin_T.yaml"],

            "program": "${workspaceFolder}/scripts/eval_origin.py",             
            "args": [""--checkpoint=/home/path.ckpt"]



[fbnet]: <https://www.mdpi.com/1424-8220/21/8/2691>
