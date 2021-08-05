# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os

#import open3d as o3d
import torch
import torch.nn.functional as funct
from glob import glob
from cv2 import imwrite

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera import Camera
#from packnet_sfm.geometry.pose import from_vec


import threading
import time

i= 0
pcd_list = []
axis_set_list = []
trajectory_list = []
fov_set_list = []


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args

def path_loader(pts,center):

    min_z = pts[:,2].min()
    max_z = pts[:,2].max()
    step = 4
    res = (max_z - min_z)/step

    path = [center]
    for i in range(0,step):
        step_in = min_z + res*(i) 
        step_out = min_z + res*(i+1)
        idx = np.where(pts[:,2]< step_out)
        subptx = pts[idx]
        idx = np.where(subptx[:,2]>=step_in)
        test = subptx[idx]
        x = test[:,0].mean()
        y = test[:,1].mean()
        z = (step_in + step_out)/2.0
        path.append([x,y,z])
    return path

def path_loader_inv(pts,center):

    min_z = pts[:,2].min()
    max_z = pts[:,2].max()
    step = 4
    res = (max_z - min_z)/step

    path = [center]
    #path = []
    for i in range(0,step):
        step_in = max_z - res*(i+1) 
        step_out = max_z - res*(i)
        idx = np.where(pts[:,2]< step_out)
        subptx = pts[idx]
        idx = np.where(subptx[:,2]>=step_in)
        test = subptx[idx]
        x = test[:,0].mean()
        y = test[:,1].mean()
        z = (step_in + step_out)/2.0
        path.append([x,y,z])
    return path

def animation_callback(vis):
    
    global pcd_list
    if len(pcd_list) >0 :
        next_pcd = pcd_list.pop(0)
        next_axis = axis_set_list.pop(0)
        next_trajectory = trajectory_list.pop(0)
        next_fov = fov_set_list.pop(0)
        print("updated - Queue count = ", len(pcd_list))
        global pcd
        pcd.colors = next_pcd.colors
        pcd.points = next_pcd.points 

        global axis_set
        axis_set.points = next_axis.points
        axis_set.lines = next_axis.lines
        axis = np.asarray(next_axis.points)
        axis_color = [[i/len(axis), i/len(axis), i/len(axis)] for i in range(len(axis))]

        global trajectory_set
        trajectory_set.points = next_trajectory.points 
        trajectory_set.lines = next_trajectory.lines 
        
        global fov_set
        fov_set.points = next_fov.points
        fov_set.lines = next_fov.lines
        #axis_set.colors=o3d.utility.Vector3dVector(np.float64(axis_color))

   
        #print(np.asarray(axis_set.points), np.asarray(axis_set.lines))
        vis.update_geometry(pcd)
        vis.update_geometry(axis_set)
        vis.update_geometry(trajectory_set)
        vis.update_geometry(fov_set)
        vis.update_renderer()
        vis.poll_events()
    else:
        time.sleep(0.03)

def update_path():
    global i
    dirname = "/home/sj/src/packnet-sfm/media/image_"
    filenames_image = os.listdir(dirname)
    filenames_image.sort()

    #dirname_depth = "/home/sj/src/open3d/"+ d_path+"_result"
    dirname_depth = "/home/sj/src/packnet-sfm/media/gt_"
    filenames_depth = os.listdir(dirname_depth)
    filenames_depth.sort()  

    full_image_filename = os.path.join(dirname, filenames_image[i])
    full_depth_filename = os.path.join(dirname_depth, filenames_depth[i])
    i=i+1
    print(i, full_image_filename)
    return full_image_filename, full_depth_filename

def load_image_my(image_path, depth_path):
    color_raw = o3d.io.read_image(image_path)
    pp =  np.concatenate((np.array(color_raw)[:,:,0].reshape(256,256,1), np.array(color_raw)[:,:,1].reshape(256,256,1), np.array(color_raw)[:,:,2].reshape(256,256,1)),axis=2)
    color_raw = o3d.geometry.Image(pp)
    depth_raw = o3d.io.read_image(depth_path)
    return color_raw, depth_raw

def set_fov_line():
    fov_center = [0,0,0]
    near = 0.12
    far = 0.5
    width = 0.25
    height = 0.25
    ratio = near/far

    fov_near_lt = [-width*ratio, height*ratio, near]
    fov_near_lb = [-width*ratio, -height*ratio, near]
    fov_near_rt = [width*ratio, height*ratio, near]
    fov_near_rb = [width*ratio, -height*ratio, near]

    fov_far_lt = [-width, height, far]
    fov_far_lb = [-width, -height, far]
    fov_far_rt = [width, height, far]
    fov_far_rb = [width, -height, far]

    fov = [ fov_near_lt, fov_near_lb,fov_near_rb , fov_near_rt, fov_far_lt,fov_far_lb,fov_far_rb,fov_far_rt, fov_center]
    fov_lines = [[0,1], [1,2],[2,3], [0,3], [4,5],[5,6],[6,7],[4,7],[4,8],[5,8],[6,8],[7,8]] 
    fov_color = [[0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]]
    return fov, fov_lines, fov_color


def run_vis():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    source_color,source_depth = update_path()

    source_color_raw, source_depth_raw = load_image_my(source_color,source_depth )
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color_raw, source_depth_raw,convert_rgb_to_intensity=False)

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    #pinhole_camera_intrinsic.set_intrinsics(256,256,200.48,250.4,128,128)
    pinhole_camera_intrinsic.set_intrinsics(256,256,148.48,148.4,128,128)

    global pcd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
    #test
    overlap_depth = 0.0002
    #min_value= min_depth(pcd)
    #pcd= point_cutting(pcd,min_value+overlap_depth)
    #~test
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    scaled = np.asarray(pcd.points)
    scaled[:,2] = scaled[:,2] * 1000
    pcd.points = o3d.utility.Vector3dVector(scaled)

    pts = np.asarray(pcd.points)

    #path 
    axis = path_loader_inv(pts,[0,0,0])
    lines = [[i, i+1] for i in range(len(axis)-1)]
    axis_color = [[i/len(lines), 0, 1-i/len(lines)] for i in range(len(lines))]
    print(axis, lines)
    global axis_set
    axis_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axis),
        lines=o3d.utility.Vector2iVector(lines)
    )    
    axis_set.colors=o3d.utility.Vector3dVector(np.float64(axis_color))

    #trajectory 
    trajectory = [[0, 0, 0], [0,0,-0.1]]
    trajectory_lines = [[i, i+1] for i in range(len(trajectory)-1)]    
    global trajectory_set
    trajectory_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory),
        lines=o3d.utility.Vector2iVector(trajectory_lines)
    )        

    #base 
    base = [ [0,0,0], [ 0,1.0,0], [1.0,0,0], [0,0,-1] ]
    base_lines = [[0,1], [0,2],[0,3]] 
    base_color = [[0,1,1],[1,0,0],[0,1,0]]
    
    base_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(base),
        lines=o3d.utility.Vector2iVector(base_lines)
    )    
    base_set.colors=o3d.utility.Vector3dVector(np.float64(base_color))

    #FOV area
    fov, fov_lines, fov_color = set_fov_line()
    global fov_set
    fov_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(fov),
        lines=o3d.utility.Vector2iVector(fov_lines)
    )    
    fov_set.colors=o3d.utility.Vector3dVector(np.float64(fov_color))

    vis.add_geometry(pcd)
    vis.add_geometry(axis_set)
    vis.add_geometry(trajectory_set)
    vis.add_geometry(base_set)
    vis.add_geometry(fov_set)

    render = vis.get_render_option()
    render.point_size = 5.0
    #render.show_coordinate_frame = True
    render.point_show_normal = True
    
    vis.register_animation_callback(animation_callback)
    vis.run()

def update_pose(PrevT, T):
    cam_to_world = np.dot( PrevT , T)
    xyzs = cam_to_world[:3, 3]
    return cam_to_world, xyzs

@torch.no_grad()
def infer_and_vis(input_file, prev_path, output_file, model_wrapper, image_shape, half, save , T, trajectory_lines_list,trajectory_points_list):

    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file).convert('RGB')
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Load image
    prev_image = load_image(prev_path).convert('RGB')
    # Resize and to tensor
    prev_image = resize_image(prev_image, image_shape)
    prev_image = to_tensor(prev_image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        prev_image = prev_image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]
    _, depth = disp_to_depth(pred_inv_depth, min_depth=0.1, max_depth=100.0)
    depth_np = depth.permute(1,2,0).detach().cpu().numpy()*255
    #     rgb = image[0].permute(1, 2, 0).detach().cpu().numpy()*255

    
    pose = model_wrapper.pose(image,prev_image)
    transPose = [Pose.from_vec(pose[:, i], 'euler') for i in range(pose.shape[1])]



    #camera intrinsic matrix
    K = np.array([[1.08, 0, 0.5, 0],
                [0, 1.02, 0.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

    #camera intrinsic matrix scaling.float()
    K[0, :] *= 256//2**0
    K[1, :] *= 256//2**0

    K = K[0:3,0:3]
    K = to_tensor(K).to('cuda')

    tcw = Pose.identity(len(K)).to('cuda')
    cam = Camera(K=K, Tcw = tcw)

    world_points = cam.reconstruct2(depth, frame='w')
    cam_points = world_points.view(3,-1).cpu().numpy()

    ref_cam = Camera(K=K, Tcw = transPose[0])
    xy, z = ref_cam.project2(world_points, frame='w')
    z= z.view(1,1, 256, 256)
    sampling_depth = funct.grid_sample(z, xy,  mode='bilinear',padding_mode='zeros', align_corners=True)
    #_, sampling_depth= disp_to_depth(sampling_depth, min_depth=0.1, max_depth=100.0)
    sampling_depth= sampling_depth.view(1,256,256)
    sampling_depth_np = sampling_depth.permute(1,2,0).detach().cpu().numpy()*255
    #sampling_depth_np.save("test.png")
    imwrite("sampled_depth.png",sampling_depth_np )
    imwrite("depth.png",depth_np )
    # z to 256,256 

    # grid sample z  using xy 

    pts = np.transpose(cam_points)            
    pts = np.float64(pts)

    #openc3d geometry format
    next_pcd = o3d.geometry.PointCloud()
    next_pcd.points = o3d.utility.Vector3dVector(pts)

    image = image.view(-1,65536).permute(1,0).cpu().numpy()
    image = np.float64(image)            
    color = o3d.utility.Vector3dVector(image)

    next_pcd.colors = color

    tt = transPose[0].to('cpu').item().numpy()[0]
    
    tt[0,3] = (tt[0,3]/5)
    tt[1,3] = (tt[1,3]/5)
    tt[2,3] = (tt[2,3]/15)

    T, xyzs = update_pose(T, tt)
    center = xyzs[0:3]
    x = xyzs[0]
    y = xyzs[1]
    z = xyzs[2] 

    next_pcd.transform(T)        

    #path 
    pts = np.asarray(next_pcd.points)
    axis = path_loader_inv(pts,center)
    lines = [[i, i+1] for i in range(len(axis)-1)]
    axis_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axis),
        lines=o3d.utility.Vector2iVector(lines),
    )

    #trajectory
    trajectory_points_list.append([x,y,z])
    trajectory_lines_list.append([len(trajectory_points_list)-2,len(trajectory_points_list)-1])

    trajectory = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory_points_list),
        lines=o3d.utility.Vector2iVector(trajectory_lines_list),
    )

    #fovq
    fov_points, fov_lines, _ = set_fov_line()
    fov = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(fov_points),
        lines=o3d.utility.Vector2iVector(fov_lines),
    )
    fov.transform(T)


    
    #add global list (Queue) (pcd, path, trajectory)
    #global pcd_list
    pcd_list.append(next_pcd)

    #global axis_set_list
    axis_set_list.append(axis_set)
    
    #global trajectory_list
    trajectory_list.append(trajectory)
    
    fov_set_list.append(fov)
    return T
    
@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half, save):

    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file).convert('RGB')
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]
    #_, depth = disp_to_depth(pred_inv_depth, min_depth=0.1, max_depth=100.0)
    depth = 1/pred_inv_depth
    #d = inv2depth(pred_inv_depth)
    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=inv2depth(pred_inv_depth))
    else:
        # Prepare RGB image
        depth_np = depth.permute(1,2,0).detach().cpu().numpy()*32
        #depth_np
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        #image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        #imwrite(output_file, depth[:, :, ::-1])
        imwrite(output_file, depth_np)
    return pred_inv_depth

@torch.no_grad()
def infer_and_save_depth_feedback(input_file, output_file, model_wrapper, image_shape, half, save , prev_image):

    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file).convert('RGB')
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Load image
    pimage = load_image(prev_image).convert('RGB')
    # Resize and to tensor
    pimage = resize_image(pimage, image_shape)
    pimage = to_tensor(pimage).unsqueeze(0)

    

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        pimage = pimage.to('cuda:{}'.format(rank()), dtype=dtype)


    # Depth inference (returns predicted inverse depth)

    #combined_input = 
    prev_inv_depth = model_wrapper.depth(pimage)[0]

    prev_inv_depth= prev_inv_depth.unsqueeze(1)
    image = torch.cat([image, prev_inv_depth] , 1)
    pred_inv_depth = model_wrapper.depth2(image)[0]
    #_, depth = disp_to_depth(pred_inv_depth, min_depth=0.1, max_depth=100.0)
    depth = 1/pred_inv_depth
    #d = inv2depth(pred_inv_depth)
    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=inv2depth(pred_inv_depth))
    else:
        # Prepare RGB image
        depth_np = depth.permute(1,2,0).detach().cpu().numpy()*32
        #depth_np
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        #image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        #imwrite(output_file, depth[:, :, ::-1])
        imwrite(output_file, depth_np)
    return pred_inv_depth



def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if os.path.isdir(args.input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [args.input]
    
    
    # t = threading.Thread(target=run_vis)
    # t.start()


   # while True:
    global pcd_list, axis_set_list, trajectory_list, fov_set_list
    pcd_list = []
    axis_set_list = []
    trajectory_list = []
    fov_set_list =[]
    trajectory_lines_list = [[0,1]]
    trajectory_points_list = [[0,0,0],[0,0,0]]
    init_pose = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    T = init_pose

    # Process each file
    i = 0
    for fn in files[rank()::world_size()]:

        if i>0 :
            prev= files[i-1]
        #if i==0:
        prev_inv_depth = infer_and_save_depth(fn, args.output, model_wrapper, image_shape, args.half, args.save)
        #else :
        #    infer_and_save_depth_feedback(fn, args.output, model_wrapper, image_shape, args.half, args.save, prev)

        i = i +1
    #for 3d vis
    #for fn in files[rank()::world_size()]:
    # for idx, image_path in enumerate(files):

    #     if idx ==0:
    #         continue
    #     prev_path = files[idx-1]
    #     T = infer_and_vis(image_path,prev_path, args.output, model_wrapper, image_shape, args.half, args.save, T,trajectory_lines_list,trajectory_points_list)


if __name__ == '__main__':
    args = parse_args()
    main(args)
