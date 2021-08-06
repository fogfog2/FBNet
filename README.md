This ropository is referenced from https://github.com/TRI-ML/packnet-sfm
"configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/train_origin.py",

            "args": [  "configs/overfit_kitti_swin_T.yaml"],
            "cwd": "${workspaceFolder}",
            "env": {"PYTHONPATH": "${cwd}" }
        }
    ]
