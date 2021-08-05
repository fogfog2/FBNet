# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as funct

########################################################################################################################

def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Construct a [3,3] camera intrinsics from pinhole parameters"""
    return torch.tensor([[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0,  1]], dtype=dtype, device=device)

def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K

########################################################################################################################

def view_synthesis(ref_image, depth, ref_cam, cam,
                   mode='bilinear', padding_mode='zeros'):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')
    # View-synthesis given the projected reference points
    return funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)

def view_synthesis_with_depth(ref_image, depth, ref_cam, cam,
                   mode='bilinear', padding_mode='zeros'):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')

    inv_ref_coords , warped_depth = ref_cam.project2(world_points, frame='c')

    # View-synthesis given the projected reference points

    ref_warped  = funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)

    warped_depth = warped_depth.view(-1,1,ref_warped.size()[2],ref_warped.size()[3])
    remapped_depth = funct.grid_sample(warped_depth, inv_ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)

    return ref_warped, remapped_depth

########################################################################################################################
def view_synthesis_with_depth_recons(ref_image, depth, ref_cam, cam, ref_depth,
                   mode='bilinear', padding_mode='zeros'):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    #depth -> Q_t
    world_points = cam.reconstruct(depth, frame='w')

    #depth -> Q_s
    prev_world_points = cam.reconstruct(ref_depth, frame='w')


    # Project world points onto reference camera
    #Q_t -> Q_t->s -> p_s 
    ref_coords = ref_cam.project(world_points, frame='w')

    #Q_s -> Q_s->t -> p_t, z_s->t
    _ , remmaped_depth = ref_cam.project_inv(prev_world_points, frame='w')

    # View-synthesis given the projected reference points
    ref_warped  = funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)


    remmaped_depth = remmaped_depth.view(-1,1,ref_warped.size()[2],ref_warped.size()[3])

    return ref_warped, remmaped_depth

########################################################################################################################


def view_synthesis_with_disp(ref_image, depth, ref_cam, cam, ref_disp,
                   mode='bilinear', padding_mode='zeros'):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')

    ref_warped  = funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)

    remapped_disp = funct.grid_sample(ref_disp, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)

    return ref_warped, remapped_disp

########################################################################################################################
