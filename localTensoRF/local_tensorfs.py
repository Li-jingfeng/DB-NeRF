# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import torch
from models.tensorBase import AlphaGridMask

from models.tensoRF import TensorVMSplit, TensorVMSplit_TimeEmbedding

from utils.utils import mtx_to_sixD, sixD_to_mtx
from utils.ray_utils import get_ray_directions_lean, get_rays_lean, get_ray_directions_360
from utils.utils import N_to_reso
from utils.renderer import raw2outputs, induce_flow
from utils.utils import TVLoss, compute_depth_loss

def ids2pixel_view(W, H, ids):
    """
    Regress pixel coordinates from ray indices
    """
    col = ids % W
    row = (ids // W) % H
    view_ids = ids // (W * H)
    return col, row, view_ids

def ids2pixel(W, H, ids):
    """
    Regress pixel coordinates from ray indices
    """
    col = ids % W
    row = (ids // W) % H
    return col, row

class LocalTensorfs(torch.nn.Module):
    """
    Self calibrating local tensorfs.
    """

    def __init__(
        self,
        fov,
        n_init_frames,
        n_overlap,
        WH,
        n_iters_per_frame,
        n_iters_reg,
        lr_R_init,
        lr_t_init,
        lr_i_init,
        lr_exposure_init,
        rf_lr_init,
        rf_lr_basis,
        lr_decay_target_ratio,
        N_voxel_list,
        update_AlphaMask_list,
        camera_prior,
        device,
        lr_upsample_reset,
        args,
        **tensorf_args,
    ):
        super(LocalTensorfs, self).__init__()
        self.args = args
        self.fov = fov
        self.n_init_frames = n_init_frames
        self.n_overlap = n_overlap
        self.W, self.H = WH
        self.n_iters_per_frame = n_iters_per_frame
        self.n_iters_reg_per_frame = n_iters_reg
        self.lr_R_init, self.lr_t_init, self.lr_i_init, self.lr_exposure_init = lr_R_init, lr_t_init, lr_i_init, lr_exposure_init
        self.rf_lr_init, self.rf_lr_basis, self.lr_decay_target_ratio = rf_lr_init, rf_lr_basis, lr_decay_target_ratio
        self.N_voxel_per_frame_list = N_voxel_list
        self.update_AlphaMask_per_frame_list = update_AlphaMask_list
        self.device = torch.device(device)
        self.camera_prior = camera_prior
        self.tensorf_args = tensorf_args
        self.is_refining = False
        self.lr_upsample_reset = lr_upsample_reset

        self.lr_factor = 1
        self.regularize = True
        self.n_iters_reg = self.n_iters_reg_per_frame
        self.n_iters = self.n_iters_per_frame
        self.update_AlphaMask_list = update_AlphaMask_list
        self.N_voxel_list = N_voxel_list

        # Setup pose and camera parameters
        self.r_c2w, self.t_c2w, self.exposure = torch.nn.ParameterList(), torch.nn.ParameterList(), torch.nn.ParameterList()
        self.r_optimizers, self.t_optimizers, self.exp_optimizers, self.pose_linked_rf = [], [], [], [] 
        self.blending_weights = torch.nn.Parameter(
            torch.ones([1, 1], device=self.device, requires_grad=False), 
            requires_grad=False,
        )
        for _ in range(n_init_frames):
            self.append_frame()

        if self.camera_prior is not None:
            focal = self.camera_prior["transforms"]["fl_x"]
            focal *= self.W / self.camera_prior["transforms"]["w"]
        else:
            fov = fov * math.pi / 180
            focal = self.W / math.tan(fov / 2) / 2
        
        self.init_focal = torch.nn.Parameter(torch.Tensor([focal]).to(self.device))

        self.focal_offset = torch.nn.Parameter(torch.ones(1, device=device))
        self.center_rel = torch.nn.Parameter(0.5 * torch.ones(2, device=device))

        if lr_i_init > 0:
            self.intrinsic_optimizer = torch.optim.Adam([self.focal_offset, self.center_rel], betas=(0.9, 0.99), lr=self.lr_i_init)


        # Setup radiance fields
        self.tensorfs = torch.nn.ParameterList()
        if self.args.use_dynamic:
            self.tensorfs_dynamic = torch.nn.ParameterList()
        self.rf_iter = []
        self.world2rf = torch.nn.ParameterList()
        self.append_rf()

    def append_rf(self, n_added_frames=1):
        self.is_refining = False
        if len(self.tensorfs) > 0:
            n_overlap = min(n_added_frames, self.n_overlap, self.blending_weights.shape[0] - 1)
            weights_overlap = 1 / n_overlap + torch.arange(
                0, 1, 1 / n_overlap
            )
            self.blending_weights.requires_grad = False
            self.blending_weights[-n_overlap :, -1] = 1 - weights_overlap
            new_blending_weights = torch.zeros_like(self.blending_weights[:, 0:1])
            new_blending_weights[-n_overlap :, 0] = weights_overlap
            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, new_blending_weights], dim=1),
                requires_grad=False,
            )
            world2rf = -self.t_c2w[-1].clone().detach()
            self.tensorfs[-1].to(torch.device("cpu"))
            torch.cuda.empty_cache()
        else:
            world2rf = torch.zeros(3, device=self.device)

        self.tensorfs.append(TensorVMSplit(device=self.device, **self.tensorf_args))
        if self.args.use_dynamic:
            self.tensorfs_dynamic.append(TensorVMSplit_TimeEmbedding(device=self.device, **self.tensorf_args))

        self.world2rf.append(world2rf.clone().detach())
        
        self.rf_iter.append(0)

        grad_vars = self.tensorfs[-1].get_optparam_groups(
            self.rf_lr_init, self.rf_lr_basis
        )
        if self.args.use_dynamic:
            grad_vars.extend(self.tensorfs_dynamic[-1].get_optparam_groups(self.rf_lr_init, self.rf_lr_basis))
        self.rf_optimizer = (torch.optim.Adam(grad_vars, betas=(0.9, 0.99)))

    def append_frame(self):
        if len(self.r_c2w) == 0:
            self.r_c2w.append(torch.eye(3, 2, device=self.device))
            self.t_c2w.append(torch.zeros(3, device=self.device))

            self.pose_linked_rf.append(0)            
        else:
            self.r_c2w.append(mtx_to_sixD(sixD_to_mtx(self.r_c2w[-1].clone().detach()[None]))[0])
            self.t_c2w.append(self.t_c2w[-1].clone().detach())

            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, self.blending_weights[-1:, :]], dim=0),
                requires_grad=False,
            )

            rf_ind = int(torch.nonzero(self.blending_weights[-1, :])[0])
            self.pose_linked_rf.append(rf_ind)
                
        self.exposure.append(torch.eye(3, 3, device=self.device))

        if self.camera_prior is not None:
            idx = len(self.r_c2w) - 1
            rel_pose = self.camera_prior["rel_poses"][idx]
            last_r_c2w = sixD_to_mtx(self.r_c2w[-1].clone().detach()[None])[0]
            self.r_c2w[-1] = last_r_c2w @ rel_pose[:3, :3]
            self.t_c2w[-1].data += last_r_c2w @ rel_pose[:3, 3]
            
        self.r_optimizers.append(torch.optim.Adam([self.r_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_R_init)) 
        self.t_optimizers.append(torch.optim.Adam([self.t_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_t_init)) 
        self.exp_optimizers.append(torch.optim.Adam([self.exposure[-1]], betas=(0.9, 0.99), lr=self.lr_exposure_init)) 

    def optimizer_step_poses_only(self, loss):
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].zero_grad()
                self.t_optimizers[idx].zero_grad()
        
        loss.backward()

        # Optimize poses
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].step()
                self.t_optimizers[idx].step()
                
    def optimizer_step(self, loss, optimize_poses):
        if self.rf_iter[-1] == 0:
            self.lr_factor = 1
            self.n_iters = self.n_iters_per_frame
            self.n_iters_reg = self.n_iters_reg_per_frame
            

        elif self.rf_iter[-1] == 1:
            n_training_frames = (self.blending_weights[:, -1] > 0).sum()
            self.n_iters = int(self.n_iters_per_frame * n_training_frames)
            self.n_iters_reg = int(self.n_iters_reg_per_frame * n_training_frames)
            self.lr_factor = self.lr_decay_target_ratio ** (1 / self.n_iters)
            self.N_voxel_list = {int(key * n_training_frames): self.N_voxel_per_frame_list[key] for key in self.N_voxel_per_frame_list}
            self.update_AlphaMask_list = [int(update_AlphaMask * n_training_frames) for update_AlphaMask in self.update_AlphaMask_per_frame_list]

        self.regularize = self.rf_iter[-1] < self.n_iters_reg

        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Poses
                if optimize_poses:
                    for param_group in self.r_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    for param_group in self.t_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    self.r_optimizers[idx].zero_grad()
                    self.t_optimizers[idx].zero_grad()
                
                # Exposure
                if self.lr_exposure_init > 0:
                    for param_group in self.exp_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    self.exp_optimizers[idx].zero_grad()

        
        
        # Intrinsics
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and 
            self.is_refining
        ):
            for param_group in self.intrinsic_optimizer.param_groups:
                param_group["lr"] *= self.lr_factor
            self.intrinsic_optimizer.zero_grad()

        # tensorfs
        self.rf_optimizer.zero_grad()

        loss.backward()

        # Optimize RFs
        self.rf_optimizer.step()
        if self.is_refining:
            for param_group in self.rf_optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * self.lr_factor

        # Increase RF resolution
        if self.rf_iter[-1] in self.N_voxel_list:
            n_voxels = self.N_voxel_list[self.rf_iter[-1]]
            reso_cur = N_to_reso(n_voxels, self.tensorfs[-1].aabb)
            self.tensorfs[-1].upsample_volume_grid(reso_cur)
            if self.args.use_dynamic:
                self.tensorfs_dynamic[-1].upsample_volume_grid(reso_cur)

            if self.lr_upsample_reset:
                print("reset lr to initial")
                grad_vars = self.tensorfs[-1].get_optparam_groups(
                    self.rf_lr_init, self.rf_lr_basis
                )
                if self.args.use_dynamic:
                    grad_vars.extend(self.tensorfs_dynamic[-1].get_optparam_groups(self.lr_init, self.lr_basis))
                self.rf_optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # Update alpha mask
        if self.rf_iter[-1] in self.update_AlphaMask_list:
            reso_mask = (self.tensorfs[-1].gridSize / 2).int()
            self.tensorfs[-1].updateAlphaMask(tuple(reso_mask))
            if self.args.use_dynamic:
                self.tensorfs_dynamic[-1].updateAlphaMask(tuple(reso_mask))

        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Optimize poses
                if optimize_poses:
                    self.r_optimizers[idx].step()
                    self.t_optimizers[idx].step()
                # Optimize exposures
                if self.lr_exposure_init > 0:
                    self.exp_optimizers[idx].step()
        
        # Optimize intrinsics
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and
            self.is_refining 
        ):
            self.intrinsic_optimizer.step()

        if self.is_refining:
            self.rf_iter[-1] += 1

        can_add_rf = self.rf_iter[-1] >= self.n_iters - 1
        return can_add_rf

    def get_cam2world(self, view_ids=None, starting_id=0):
        if view_ids is not None:
            r_c2w = torch.stack([self.r_c2w[view_id] for view_id in view_ids], dim=0)
            t_c2w = torch.stack([self.t_c2w[view_id] for view_id in view_ids], dim=0)
        else:
            r_c2w = torch.stack(list(self.r_c2w[starting_id:]), dim=0)
            t_c2w = torch.stack(list(self.t_c2w[starting_id:]), dim=0)
        return torch.cat([sixD_to_mtx(r_c2w), t_c2w[..., None]], dim = -1)

    def get_kwargs(self):
        kwargs = {
            "camera_prior": None,
            "fov": self.fov,
            "n_init_frames": self.n_init_frames,
            "n_overlap": self.n_overlap,
            "WH": (self.W, self.H),
            "n_iters_per_frame": self.n_iters_per_frame,
            "n_iters_reg": self.n_iters_reg_per_frame,
            "lr_R_init": self.lr_R_init,
            "lr_t_init": self.lr_t_init,
            "lr_i_init": self.lr_i_init,
            "lr_exposure_init": self.lr_exposure_init,
            "rf_lr_init": self.rf_lr_init,
            "rf_lr_basis": self.rf_lr_basis,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "N_voxel_list": self.N_voxel_per_frame_list,
            "update_AlphaMask_list": self.update_AlphaMask_per_frame_list,
            "lr_upsample_reset": self.lr_upsample_reset,
        }
        kwargs.update(self.tensorfs[0].get_kwargs())
        if self.args.use_dynamic:
            kwargs.update(self.tensorfs_dynamic[0].get_kwargs())    

        return kwargs

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        torch.save(ckpt, path)

    def load(self, state_dict):
        # TODO A bit hacky?
        import re
        n_frames = 0
        for key in state_dict:
            if re.fullmatch(r"r_c2w.[0-9]*", key):
                n_frames += 1
            if re.fullmatch(r"tensorfs.[1-9][0-9]*.density_plane.0", key):
                self.tensorf_args["gridSize"] = [state_dict[key].shape[2], state_dict[key].shape[3], state_dict[f"{key[:-15]}density_line.0"].shape[2]]
                self.append_rf()

        for i in range(len(self.tensorfs)):
            if f"tensorfs.{i}.alphaMask.aabb" in state_dict:
                alpha_volume = state_dict[f'tensorfs.{i}.alphaMask.alpha_volume'].to(self.device)
                aabb = state_dict[f'tensorfs.{i}.alphaMask.aabb'].to(self.device)
                self.tensorfs[i].alphaMask = AlphaGridMask(self.device, aabb, alpha_volume)
                if self.args.use_dynamic:
                    self.tensorfs_dynamic[i].alphaMask = AlphaGridMask(self.device, aabb, alpha_volume)


        for _ in range(n_frames - len(self.r_c2w)):
            self.append_frame()
        
        self.blending_weights = torch.nn.Parameter(
            torch.ones_like(state_dict["blending_weights"]), requires_grad=False
        )

        self.load_state_dict(state_dict)

    def get_dist_to_last_rf(self):
        return torch.norm(self.t_c2w[-1] + self.world2rf[-1])

    def get_reg_loss(self, tvreg, TV_weight_density, TV_weight_app, L1_weight_inital):
        tv_loss = 0
        l1_loss = 0
        if self.rf_iter[-1] < self.n_iters:
            if TV_weight_density > 0:
                tv_weight = TV_weight_density * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_density(tvreg).mean() * tv_weight
                
            if TV_weight_app > 0:
                tv_weight = TV_weight_app * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_app(tvreg).mean() * tv_weight
    
            if L1_weight_inital > 0:
                l1_loss += self.tensorfs[-1].density_L1() * L1_weight_inital
        return tv_loss, l1_loss

    def focal(self, W):
        return self.init_focal * self.focal_offset * W / self.W 
    def center(self, W, H):
        return torch.Tensor([W, H]).to(self.center_rel) * self.center_rel
    
    def add_dynamic_forward(
        self,
        iteration, # iter
        tensorf_static,
        tensorf_dynamic,
        rgb_train, # rgb监督
        flow_f_train, # 前向光流监督 get from dataset
        flow_mask_f_train, # flowmask
        flow_b_train,
        flow_mask_b_train,
        v_ref,# 列
        u_ref,# 横
        allforegroundmasks_train, # mask
        focal_refine, # focal
        t_interval, # 时间间隔
        Temp,
        Temp_static,
        Temp_disp_TV,
        H,
        W,
        loss_weights,
        grid_train,
        
        view_ids,# image_id
        rays,
        ts_train,
        is_train,
        white_bg,
        N_samples,
        refine,
        floater_thresh,
        ):
        total_loss=0
        monodepth_weight_static=0.04
        monodepth_weight_dynamic=0.04
        tvreg = TVLoss()
        if view_ids==0:
            allposes_refine_f_train = self.get_cam2world(view_ids)
        else:
            allposes_refine_f_train = self.get_cam2world(view_ids-1)
        if view_ids==len(self.r_c2w)-1: 
            allposes_refine_b_train = self.get_cam2world(view_ids)
        else:
            allposes_refine_b_train = self.get_cam2world(view_ids+1)
        _, _, _, _, _, _, rgb_points_static, sigmas_static, _, _ = tensorf_static(
            rays.detach(),
            ts_train,
            None,
            is_train=True,
            white_bg=white_bg,
            N_samples=-1,
            refine=self.is_refining,
            floater_thresh=floater_thresh,
        )
        # dynamic tensorf
        (
            _,
            _,
            blending,
            pts_ref,
            _,
            _,
            rgb_points_dynamic,
            sigmas_dynamic,
            z_vals_dynamic,
            dists_dynamic,
        ) = tensorf_dynamic(
            rays.detach(),
            ts_train,
            None,
            is_train=True,
            white_bg=white_bg,
            N_samples=-1,
        )
        (
            rgb_map_full,
            _,
            _,
            _,
            rgb_map_s,
            depth_map_s,
            _,
            weights_s,
            rgb_map_d,
            depth_map_d,
            _,
            weights_d,
            dynamicness_map,
        ) = raw2outputs(
            rgb_points_static.detach(),
            sigmas_static.detach(),
            rgb_points_dynamic,
            sigmas_dynamic,
            dists_dynamic,
            blending,
            z_vals_dynamic,
            rays.detach(),
            is_train=True,
            ray_type="contract",
        )
        loss_rgb = 0.25 * ((torch.abs(rgb_map_full - rgb_train)) * loss_weights) / loss_weights.mean()
        # loss_rgb = torch.mean((rgb_map_full - rgb_train) ** 2)
        total_loss += loss_rgb
        img_d_loss = 0.25 * ((torch.abs(rgb_map_d - rgb_train)) * loss_weights) / loss_weights.mean()
        # img_d_loss = torch.mean((rgb_map_d - rgb_train) ** 2)
        total_loss += 1.0*img_d_loss

        scene_flow_f, scene_flow_b = tensorf_dynamic.get_forward_backward_scene_flow(
            pts_ref, ts_train.to(self.device)
        )
        # Flow grouping loss
        # mask loss
        if iteration >= 2000:
            mask_loss = torch.mean(
                torch.abs(dynamicness_map - allforegroundmasks_train[..., 0])
            )
            total_loss += 0.1 * mask_loss * Temp_disp_TV
        if iteration >= 10000:
            # skewed mask loss
            clamped_mask = torch.clamp(dynamicness_map, min=1e-6, max=1.0 - 1e-6)
            skewed_mask_loss = torch.mean(
                -(
                    (clamped_mask**2) * torch.log((clamped_mask**2))
                    + (1 - (clamped_mask**2)) * torch.log(1 - (clamped_mask**2))
                )
            )
            total_loss += 0.01 * skewed_mask_loss

            mask_L1_reg_loss = torch.mean(torch.abs(dynamicness_map))
            total_loss += 0.01 * mask_L1_reg_loss
        # forward backward flow
        pts_f = torch.clamp(pts_ref + scene_flow_f, min=-2.0 + 1e-6, max=2.0 - 1e-6)
        pts_b = torch.clamp(pts_ref + scene_flow_b, min=-2.0 + 1e-6, max=2.0 - 1e-6)
        
        induced_flow_f, induced_disp_f = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_f_train.detach(),# pose
            weights_d,
            pts_f,
            grid_train,
            rays.detach(),
            ray_type="contract",
        )
        flow_f_loss = (
            torch.sum(torch.abs(induced_flow_f - flow_f_train) * flow_mask_f_train)
            / (torch.sum(flow_mask_f_train) + 1e-8)
            / flow_f_train.shape[-1]
        )
        total_loss += 0.02 * flow_f_loss * Temp
        induced_flow_b, induced_disp_b = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_b_train.detach(),
            weights_d,
            pts_b,
            grid_train,
            rays.detach(),
            ray_type="contract",
        )
        flow_b_loss = (
            torch.sum(torch.abs(induced_flow_b - flow_b_train) * flow_mask_b_train)
            / (torch.sum(flow_mask_b_train) + 1e-8)
            / flow_b_train.shape[-1]
        )
        total_loss += 0.02 * flow_b_loss * Temp

        # disparity loss
        # forward
        uv_f = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_f_train.device)
            + flow_f_train
        )
        directions_f = torch.stack(
            [
                (uv_f[..., 0] - W / 2) / (focal_refine.detach()),
                -(uv_f[..., 1] - H / 2) / (focal_refine.detach()),
                -torch.ones_like(uv_f[..., 0]),
            ],
            -1,
        )  # (H, W, 3)
        rays_f_o, rays_f_d = get_rays_lean(directions_f, allposes_refine_f_train)
        rays_f_train = torch.cat([rays_f_o, rays_f_d], -1).view(-1, 6)
        _, _, _, _, _, _, rgb_points_static_f, sigmas_static_f, _, _ = tensorf_static(
            rays_f_train.detach(),
            ts_train + t_interval,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )
        (
            _,
            _,
            blending_f,
            pts_ref_ff,
            _,
            _,
            rgb_points_dynamic_f,
            sigmas_dynamic_f,
            z_vals_dynamic_f,
            dists_dynamic_f,
        ) = tensorf_dynamic(
            rays_f_train.detach(),
            ts_train + t_interval,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )
        _, _, _, _, _, _, _, _, _, _, _, weights_d_f, _ = raw2outputs(
            rgb_points_static_f.detach(),
            sigmas_static_f.detach(),
            rgb_points_dynamic_f,
            sigmas_dynamic_f,
            dists_dynamic_f,
            blending_f,
            z_vals_dynamic_f,
            rays_f_train.detach(),
            is_train=True,
            ray_type="contract",
        )
        _, induced_disp_ff = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_f_train.detach(),
            weights_d_f,
            pts_ref_ff,
            grid_train,
            rays_f_train.detach(),
            ray_type="contract",
        )
        disp_f_loss = torch.sum(
            torch.abs(induced_disp_f - induced_disp_ff) * flow_mask_f_train
        ) / (torch.sum(flow_mask_f_train) + 1e-8)
        total_loss += 0.04 * disp_f_loss * Temp
        # backward
        uv_b = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_b_train.device)
            + flow_b_train
        )
        directions_b = torch.stack(
            [
                (uv_b[..., 0] - W / 2) / (focal_refine.detach()),
                -(uv_b[..., 1] - H / 2) / (focal_refine.detach()),
                -torch.ones_like(uv_b[..., 0]),
            ],
            -1,
        )  # (H, W, 3)
        rays_b_o, rays_b_d = get_rays_lean(directions_b, allposes_refine_b_train)
        rays_b_train = torch.cat([rays_b_o, rays_b_d], -1).view(-1, 6)
        _, _, _, _, _, _, rgb_points_static_b, sigmas_static_b, _, _ = tensorf_static(
            rays_b_train.detach(),
            ts_train - t_interval,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )
        (
            _,
            _,
            blending_b,
            pts_ref_bb,
            _,
            _,
            rgb_points_dynamic_b,
            sigmas_dynamic_b,
            z_vals_dynamic_b,
            dists_dynamic_b,
        ) = tensorf_dynamic(
            rays_b_train.detach(),
            ts_train - t_interval,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )
        _, _, _, _, _, _, _, _, _, _, _, weights_d_b, _ = raw2outputs(
            rgb_points_static_b.detach(),
            sigmas_static_b.detach(),
            rgb_points_dynamic_b,
            sigmas_dynamic_b,
            dists_dynamic_b,
            blending_b,
            z_vals_dynamic_b,
            rays_b_train.detach(),
            is_train=True,
            ray_type="contract",
        )
        _, induced_disp_bb = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_b_train.detach(),
            weights_d_b,
            pts_ref_bb,
            grid_train,
            rays_b_train.detach(),
            ray_type="contract",
        )
        disp_b_loss = torch.sum(
            torch.abs(induced_disp_b - induced_disp_bb) * flow_mask_b_train
        ) / (torch.sum(flow_mask_b_train) + 1e-8)
        total_loss += 0.04 * disp_b_loss * Temp

        smooth_scene_flow_loss = torch.mean(torch.abs(scene_flow_f + scene_flow_b))
        total_loss += smooth_scene_flow_loss * 0.1

        # monocular depth loss
        # total_mono_depth_loss += compute_depth_loss(
        #                 1.0 / (depth_map_d[valid] + 1e-6), alldisps_train[valid]
        #             )
        # total_loss += total_mono_depth_loss * monodepth_weight_dynamic * Temp
        # adaptive Order loss
        order_loss = torch.sum(
                (
                    (1.0 / (depth_map_d + 1e-6) - 1.0 / (depth_map_s.detach() + 1e-6))
                    ** 2
                )
                * (1.0 - dynamicness_map.detach())
            ) / (torch.sum((1.0 - dynamicness_map.detach())) + 1e-8)
        total_loss += order_loss * 10.0
        
        # static part for pose estimation
        # static tensorf
        (
            _,
            _,
            _,
            pts_ref_s,
            _,
            _,
            rgb_points_static,
            sigmas_static,
            _,
            _,
        ) = tensorf_static(
            rays,
            ts_train,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )
        # dynamic tensorf
        (
            _,
            _,
            blending,
            pts_ref,
            _,
            _,
            rgb_points_dynamic,
            sigmas_dynamic,
            z_vals_dynamic,
            dists_dynamic,
        ) = tensorf_dynamic(
            rays,
            ts_train,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )

        _, _, _, _, rgb_map_s, depth_map_s, _, weights_s, _, _, _, _, _ = raw2outputs(
            rgb_points_static,
            sigmas_static,
            rgb_points_dynamic,
            sigmas_dynamic,
            dists_dynamic,
            blending,
            z_vals_dynamic,
            rays,
            is_train=True,
            ray_type="contract",
        )
        ### static losses
        # RGB loss
        img_s_loss = (
            torch.sum(
                (rgb_map_s - rgb_train) ** 2
                * (1.0 - allforegroundmasks_train[..., 0:1])
            )
            / (torch.sum((1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8)
            / rgb_map_s.shape[-1]
        )
        total_loss += 0.25 * img_s_loss
        # optimize pose
        induced_flow_f_s, induced_disp_f_s = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_f_train,
                weights_s,
                pts_ref_s,
                grid_train,
                rays,
                ray_type="contract",
            )
        flow_f_s_loss = (
            torch.sum(
                torch.abs(induced_flow_f_s - flow_f_train)
                * flow_mask_f_train
                * (1.0 - allforegroundmasks_train[..., 0:1])
            )
            / (
                torch.sum(
                    flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])
                )
                + 1e-8
            )
            / flow_f_train.shape[-1]
        )
        total_loss += 0.02 * flow_f_s_loss * Temp_static
        induced_flow_b_s, induced_disp_b_s = induce_flow(
            H,
            W,
            focal_refine,
            allposes_refine_b_train,
            weights_s,
            pts_ref_s,
            grid_train,
            rays,
            ray_type="contract",
        )
        flow_b_s_loss = (
            torch.sum(
                torch.abs(induced_flow_b_s - flow_b_train)
                * flow_mask_b_train
                * (1.0 - allforegroundmasks_train[..., 0:1])
            )
            / (
                torch.sum(
                    flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])
                )
                + 1e-8
            )
            / flow_b_train.shape[-1]
        )
        total_loss += 0.02 * flow_b_s_loss * Temp_static
        # static disparity loss
        # forward
        uv_f = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_f_train.device)
            + flow_f_train
        )
        directions_f = torch.stack(
            [
                (uv_f[..., 0] - W / 2) / (focal_refine),
                -(uv_f[..., 1] - H / 2) / (focal_refine),
                -torch.ones_like(uv_f[..., 0]),
            ],
            -1,
        )  # (H, W, 3)
        rays_f_o, rays_f_d = get_rays_lean(
            directions_f, allposes_refine_f_train
        )  # both (b, 3)
        rays_f_train = torch.cat([rays_f_o, rays_f_d], -1).view(-1, 6)

        _, _, _, pts_ref_s_ff, weights_s_ff, _, _, _, _, _ = tensorf_static(
            rays_f_train,
            ts_train,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )
        _, induced_disp_s_ff = induce_flow(
            H,
            W,
            focal_refine,
            allposes_refine_f_train,
            weights_s_ff,
            pts_ref_s_ff,
            grid_train,
            rays_f_train,
            ray_type="contract",
        )
        disp_f_s_loss = torch.sum(
            torch.abs(induced_disp_f_s - induced_disp_s_ff)
            * flow_mask_f_train
            * (1.0 - allforegroundmasks_train[..., 0:1])
        ) / (
            torch.sum(
                flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])
            )
            + 1e-8
        )
        total_loss += 0.04 * disp_f_s_loss * Temp_static
        # backward
        uv_b = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_b_train.device)
            + flow_b_train
        )
        directions_b = torch.stack(
            [
                (uv_b[..., 0] - W / 2) / (focal_refine),
                -(uv_b[..., 1] - H / 2) / (focal_refine),
                -torch.ones_like(uv_b[..., 0]),
            ],
            -1,
        )  # (H, W, 3)
        rays_b_o, rays_b_d = get_rays_lean(
            directions_b, allposes_refine_b_train
        )  # both (b, 3)
        rays_b_train = torch.cat([rays_b_o, rays_b_d], -1).view(-1, 6)
        _, _, _, pts_ref_s_bb, weights_s_bb, _, _, _, _, _ = tensorf_static(
            rays_b_train,
            ts_train,
            None,
            is_train=True,
            white_bg=white_bg,
            ray_type="contract",
            N_samples=-1,
        )
        _, induced_disp_s_bb = induce_flow(
            H,
            W,
            focal_refine,
            allposes_refine_b_train,
            weights_s_bb,
            pts_ref_s_bb,
            grid_train,
            rays_b_train,
            ray_type="contract",
        )
        disp_b_s_loss = torch.sum(
            torch.abs(induced_disp_b_s - induced_disp_s_bb)
            * flow_mask_b_train
            * (1.0 - allforegroundmasks_train[..., 0:1])
        ) / (
            torch.sum(
                flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])
            )
            + 1e-8
        )
        total_loss += 0.04 * disp_b_s_loss * Temp_static

        return total_loss, depth_map_d, depth_map_s
        # Monocular depth loss with mask for static TensoRF
        # total_mono_depth_loss += compute_depth_loss(
        #                     1.0 / (depth_map_s[valid] + 1e-6), alldisps_train[valid]
        #                 )
        # counter += torch.sum(valid)
        # total_mono_depth_loss = total_mono_depth_loss / counter
        # total_loss += (
        #     total_mono_depth_loss * monodepth_weight_static * Temp_static
        # )

    def forward(
        self,
        ray_ids,
        view_ids,
        W,
        H,
        ts=None,
        white_bg=True,
        is_train=True,
        cam2world=None,
        world2rf=None,
        blending_weights=None,
        chunk=16384,
        test_id=False,
        floater_thresh=0,
    ):
        i, j = ids2pixel(W, H, ray_ids)
        if self.fov == 360:
            directions = get_ray_directions_360(i, j, W, H)
        else:
            directions = get_ray_directions_lean(i, j, self.focal(W), self.center(W, H))

        if blending_weights is None:
            blending_weights = self.blending_weights[view_ids].clone()
        if cam2world is None:
            cam2world = self.get_cam2world(view_ids)
        if world2rf is None:
            world2rf = self.world2rf

        # Train a single RF at a time
        if is_train:
            blending_weights[:, -1] = 1
            blending_weights[:, :-1] = 0

        if is_train:
            active_rf_ids = [len(self.tensorfs) - 1]
        else:
            active_rf_ids = torch.nonzero(torch.sum(blending_weights, dim=0))[:, 0].tolist()
        ij = torch.stack([i, j], dim=-1)
        if len(active_rf_ids) == 0:
            print("****** No valid RF")
            return torch.ones([ray_ids.shape[0], 3]), torch.ones_like(ray_ids).float(), torch.ones_like(ray_ids).float(), directions, ij

        cam2rfs = {}
        initial_devices = []
        for rf_id in active_rf_ids:
            cam2rf = cam2world.clone()
            cam2rf[:, :3, 3] += world2rf[rf_id]

            cam2rfs[rf_id] = cam2rf
            
            initial_devices.append(self.tensorfs[rf_id].device)
            if initial_devices[-1] != view_ids.device:
                self.tensorfs[rf_id].to(view_ids.device)

        for key in cam2rfs:
            cam2rfs[key] = cam2rfs[key].repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        blending_weights_expanded = blending_weights.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        rgbs_static = torch.zeros_like(directions)
        depth_maps_static = torch.zeros_like(directions[..., 0])
        rgbs_dynamic = torch.zeros_like(directions)
        depth_maps_dynamic = torch.zeros_like(directions[..., 0])
        rgbs = torch.zeros_like(directions) 
        depth_maps = torch.zeros_like(directions[..., 0]) 
        mask = torch.zeros_like(directions[..., 0])
        N_rays_all = ray_ids.shape[0]
        chunk = chunk // len(active_rf_ids)
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            if chunk_idx != 0:
                torch.cuda.empty_cache()
            directions_chunk = directions[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            blending_weights_chunk = blending_weights_expanded[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ]
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk] if ts is not None else None

            for rf_id in active_rf_ids:
                blending_weight_chunk = blending_weights_chunk[:, rf_id]
                cam2rf = cam2rfs[rf_id][chunk_idx * chunk : (chunk_idx + 1) * chunk]

                rays_o, rays_d = get_rays_lean(directions_chunk, cam2rf)
                rays = torch.cat([rays_o, rays_d], -1).view(-1, 6)
                # 靠着这个学习pose
                _, _, blending_s, pts_ref_s, weight_s, xyz_prime_s, rgb_s, sigma_s, z_vals_s, dists_s, rgb_map_t, depth_map_t = self.tensorfs[rf_id](
                    rays,
                    ts_chunk=None,
                    is_train=is_train,
                    white_bg=white_bg,
                    N_samples=-1,
                    refine=self.is_refining,
                    floater_thresh=floater_thresh,
                )
                if self.args.use_dynamic:
                    _, _, blending_d, pts_ref_d, weight_d, xyz_prime_d, rgb_d, sigma_d, z_vals_d, dists_d, _, _ = self.tensorfs_dynamic[rf_id](
                        rays.detach(),
                        ts_chunk=ts_chunk,
                        timeembeddings_chunk=None,
                        is_train=is_train,
                        white_bg=white_bg,
                        N_samples=-1,
                        refine=self.is_refining,
                        floater_thresh=floater_thresh,
                    )
                    (
                        rgb_map_full,
                        depth_map_full,
                        _,
                        _,
                        rgb_map_s,
                        depth_map_s,
                        _,
                        weights_s,
                        rgb_map_d,
                        depth_map_d,
                        _,
                        weights_d,
                        dynamicness_map,
                    )= raw2outputs(rgb_s, sigma_s, rgb_d, sigma_d, dists_d, blending_d, z_vals_d, rays, is_train=True, ray_type="contract")
                # rgb_map_s和rgb_map_t是否一致
                if self.args.use_dynamic:
                    rgbs_static[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        rgbs_static[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        rgb_map_s * blending_weight_chunk[..., None]
                    )
                    depth_maps_static[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        depth_maps_static[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        depth_map_s * blending_weight_chunk
                    )
                    rgbs_dynamic[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        rgbs_dynamic[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        rgb_map_d * blending_weight_chunk[..., None]
                    )
                    depth_maps_dynamic[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        depth_maps_dynamic[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        depth_map_d * blending_weight_chunk
                    )
                    rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        rgb_map_full * blending_weight_chunk[..., None]
                    )
                    depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        depth_map_full * blending_weight_chunk
                    )
                    # mask
                    mask[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        mask[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        dynamicness_map * blending_weight_chunk
                    )
                else:
                    rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        rgb_map_t * blending_weight_chunk[..., None]
                    )
                    depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                        depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                        depth_map_t * blending_weight_chunk
                    )

        for rf_id, initial_device in zip(active_rf_ids, initial_devices):
            if initial_device != view_ids.device:
                self.tensorfs[rf_id].to(initial_device)
                torch.cuda.empty_cache()

        if self.lr_exposure_init > 0:
            # TODO: cleanup
            if test_id:
                view_ids_m = torch.maximum(view_ids - 1, torch.tensor(0, device=view_ids.device))
                view_ids_m[view_ids_m==view_ids] = 1
                
                view_ids_p = torch.minimum(view_ids + 1, torch.tensor(len(self.exposure) - 1, device=view_ids.device))
                view_ids_p[view_ids_m==view_ids] = len(self.exposure) - 2
                
                exposure_stacked = torch.stack(list(self.exposure), dim=0).clone().detach()
                exposure = (exposure_stacked[view_ids_m] + exposure_stacked[view_ids_p]) / 2  
            else:
                exposure = torch.stack(list(self.exposure), dim=0)[view_ids]
                
            exposure = exposure.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
            rgbs = torch.bmm(exposure, rgbs[..., None])[..., 0]
            if self.args.use_dynamic:
                rgbs_dynamic = torch.bmm(exposure, rgbs_dynamic[..., None])[..., 0]
                rgbs_static = torch.bmm(exposure, rgbs_static[..., None])[..., 0]
                rgbs_dynamic = rgbs_dynamic.clamp(0, 1)
                rgbs_static = rgbs_static.clamp(0, 1)
            else:
                rgbs_dynamic=None
                rgbs_static=None
        rgbs = rgbs.clamp(0, 1)

        return rgbs, depth_maps, rgbs_static, depth_maps_static, rgbs_dynamic, depth_maps_dynamic, mask, directions, ij