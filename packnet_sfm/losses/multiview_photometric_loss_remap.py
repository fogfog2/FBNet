# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_utils import view_synthesis, view_synthesis_with_depth , view_synthesis_with_disp, view_synthesis_with_depth_recons
from packnet_sfm.utils.depth import calc_smoothness, inv2depth, depth2inv
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling

########################################################################################################################

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

########################################################################################################################

class MultiViewPhotometricLossRemap(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scalesto consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_scales=4, ssim_loss_weight=0.85, occ_reg_weight=0.1, smooth_loss_weight=0.1,
                 C1=1e-4, C2=9e-4, photometric_reduce_op='mean', disp_norm=True, clip_loss=0.5,
                 progressive_scaling=0.0, padding_mode='zeros',
                 automask_loss=False, depth_reconstruction_weight=0.05, depth_reconstruction_ssim_weight=0.15 , **kwargs):
        super().__init__()
        self.n = num_scales
        self.progressive_scaling = progressive_scaling
        self.ssim_loss_weight = ssim_loss_weight
        self.occ_reg_weight = occ_reg_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss
        self.depth_reconstruction_weight = depth_reconstruction_weight
        self.depth_reconstruction_ssim_weight = depth_reconstruction_ssim_weight
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

########################################################################################################################

    def warp_ref_image(self, inv_depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [view_synthesis(
            ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode) for i in range(self.n)]
        # Return warped reference image
        return ref_warped
########################################################################################################################

    def warp_ref_image_and_depth(self, inv_depths, ref_image, K, ref_K, pose ):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation
reduce_functionnsor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)

        ref_warped = []
        remap_depth = []
        for i in range(self.n):
             
             warped_ref, depth_remap = view_synthesis_with_depth(ref_images[i], depths[i], ref_cams[i], cams[i],  padding_mode=self.padding_mode)
             ref_warped.append(warped_ref)
             remap_depth.append(depth_remap)

        # ref_warped, remap_depth = [view_synthesis_with_depth(
        #     ref_images[i], depths[i], ref_cams[i], cams[i],
        #     padding_mode=self.padding_mode) for i in range(self.n)]

        return ref_warped, remap_disp


########################################################################################################################

    def warp_ref_image_and_disp(self, inv_depths, ref_image, K, ref_K, pose , ref_disp):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)

        ref_warped = []
        remap_disp = []
        for i in range(self.n):
             #warped_ref, depth_remap = view_synthesis_with_depth(ref_images[i], depths[i], ref_cams[i], cams[i],padding_mode=self.padding_mode)
             warped_ref, disp_remap = view_synthesis_with_disp(ref_images[i], depths[i], ref_cams[i], cams[i], ref_disp[i], padding_mode=self.padding_mode)
             ref_warped.append(warped_ref)
             remap_disp.append(disp_remap)

        # ref_warped, remap_depth = [view_synthesis_with_depth(
        #     ref_images[i], depths[i], ref_cams[i], cams[i],
        #     padding_mode=self.padding_mode) for i in range(self.n)]

        return ref_warped, remap_disp

########################################################################################################################

    def warp_ref_image_and_depth_recons(self, inv_depths, ref_image, K, ref_K, pose , ref_disp):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        prev_depths = [inv2depth(ref_disp[i]) for i in range(self.n)]

        ref_images = match_scales(ref_image, inv_depths, self.n)

        ref_warped = []
        remap_disp = []
        for i in range(self.n):             
             warped_ref, depth_remap = view_synthesis_with_depth_recons(ref_images[i], depths[i], ref_cams[i], cams[i], prev_depths[i], padding_mode=self.padding_mode)
             ref_warped.append(warped_ref)

             disp_remap = depth2inv(depth_remap)
             remap_disp.append(disp_remap)

        # ref_warped, remap_depth = [view_synthesis_with_depth(
        #     ref_images[i], depths[i], ref_cams[i], cams[i],
        #     padding_mode=self.padding_mode) for i in range(self.n)]

        return ref_warped, remap_disp

########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]

        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def calc_photometric_loss_with_disp(self, t_est, images, prev_disp, remap_disp):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]

        l1_loss_remap = [torch.abs(prev_disp[i] - remap_disp[i])
                   for i in range(self.n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True) + 0.10 * l1_loss_remap[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                minvalue = torch.cat(losses, 1).min(1, True)[0]
                mask     = torch.cat(losses, 1).min(1, True)[1]
                mask_multy = minvalue* (1-mask)
                return mask_multy.mean()
                #return torch.cat(losses, 1).min(1, True)[0].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        self.add_metric('photometric_loss', photometric_loss)
        return photometric_loss

########################################################################################################################

    def calc_depth_matching_loss(self, prev_depth, remap_depth, depth_loss_weight):
        l1_loss_remap = [torch.abs(prev_depth[i] - remap_depth[i])
                   for i in range(self.n)]

        if self.depth_reconstruction_ssim_weight > 0.0:
            ssim_loss = [self.SSIM(prev_depth[i], remap_depth[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            depth_photometric_loss = [self.depth_reconstruction_ssim_weight * ssim_loss[i].mean(1, True) +
                                (1-self.depth_reconstruction_ssim_weight) * l1_loss_remap[i].mean(1, True)
                                for i in range(self.n)]
        else:
            depth_photometric_loss = l1_loss_remap


        depthmetric_loss = [depth_photometric_loss[i].mean(1, True)
                                for i in range(self.n)]

        depthmetric_loss = sum([depthmetric_loss[i].mean()
                                for i in range(self.n)]) / self.n

        depthmetric_loss = depthmetric_loss * depth_loss_weight
        
    
        self.add_metric('depth_loss', depthmetric_loss)
        return depthmetric_loss
########################################################################################################################

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)
        return smoothness_loss

########################################################################################################################

    def forward(self, image, context, inv_depths,
                K, ref_K, poses, prev_inv_depths, return_logs=False, progress=0.0):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        context : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        poses : list of Pose
            Camera transformation between original and context
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Loop over all reference images
        photometric_losses = [[] for _ in range(self.n)]
        depth_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)
        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            ref_warped, remap_depth = self.warp_ref_image_and_depth_recons(inv_depths, ref_image, K, ref_K, pose, prev_inv_depths)
            
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i])
        # Calculate reduced photometric loss
        loss = self.reduce_photometric_loss(photometric_losses)

        #loss += self.calc_depth_matching_loss(remap_disp, prev_inv_depths, self.smooth_loss_weight)
        if self.depth_reconstruction_weight > 0.0:
            loss += self.calc_depth_matching_loss(remap_depth, inv_depths, self.depth_reconstruction_weight)

        # Include smoothness loss if requested
        #if self.smooth_loss_weight > 0.0:
        loss += self.calc_smoothness_loss(inv_depths, images)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################
