
import os
import warnings

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys
import time
import loss

from torch.utils.tensorboard import SummaryWriter

sys.path.append("localTensoRF")
from dataLoader.localrf_dataset import LocalRFDataset
from local_tensorfs import LocalTensorfs
from opt import config_parser
from renderer import render
from utils.utils import (get_fwd_bwd_cam2cams, smooth_poses_spline)
from utils.utils import (N_to_reso, TVLoss, draw_poses, get_pred_flow,
                         compute_depth_loss)
import builders
from third_party.nerfacc_prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from models.render_utils import render_rays
from omegaconf import OmegaConf
from models.radiance_field import RadianceField, DensityField
from omegaconf import OmegaConf
from dataLoader.base import SceneDataset
from models.video_utils import render_pixels, save_videos
import argparse
from typing import List, Optional

render_keys = [
    "gt_rgbs",
    "rgbs",
    "depths",
    # "median_depths",
    "gt_dino_feats",
    "dino_feats",
    "dynamic_rgbs",
    "dynamic_depths",
    "static_rgbs",
    "static_depths",
    "forward_flows",
    "backward_flows",
    "dynamic_rgb_on_static_dinos",
    "dino_pe",
    "dino_feats_pe_free",
    # "dynamic_dino_on_static_rgbs",
    # "shadow_reduced_static_rgbs",
    # "shadow_only_static_rgbs",
    # "shadows",
    # "gt_sky_masks",
    # "sky_masks",
]

def setup(args):
    # ------ get config from args -------- #
    default_config = OmegaConf.create(OmegaConf.load("configs/default_config.yaml"))
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_config, cfg)
    # log_dir = os.path.join(args.output_root, args.run_name)
    # cfg.log_dir = log_dir
    cfg.nerf.model.num_cams = cfg.data.pixel_source.num_cams
    cfg.nerf.model.unbounded = cfg.nerf.unbounded
    cfg.nerf.propnet.unbounded = cfg.nerf.unbounded
    cfg.nerf.model.resume_from = cfg.resume_from
    return cfg

def reconstruction(args):
    # Apply speedup factors
    args.n_iters_per_frame = int(args.n_iters_per_frame / args.refinement_speedup_factor)
    args.n_iters_reg = int(args.n_iters_reg / args.refinement_speedup_factor)
    args.upsamp_list = [int(upsamp / args.refinement_speedup_factor) for upsamp in args.upsamp_list]
    args.update_AlphaMask_list = [int(update_AlphaMask / args.refinement_speedup_factor) 
                                  for update_AlphaMask in args.update_AlphaMask_list]
    
    args.add_frames_every = int(args.add_frames_every / args.prog_speedup_factor)
    args.lr_R_init = args.lr_R_init * args.prog_speedup_factor
    args.lr_t_init = args.lr_t_init * args.prog_speedup_factor
    args.loss_flow_weight_inital = args.loss_flow_weight_inital * args.prog_speedup_factor
    args.L1_weight = args.L1_weight * args.prog_speedup_factor
    args.TV_weight_density = args.TV_weight_density * args.prog_speedup_factor
    args.TV_weight_app = args.TV_weight_app * args.prog_speedup_factor

    # add loss
    if cfg.data.pixel_source.load_rgb:
        rgb_loss_fn = loss.RealValueLoss(
            loss_type=cfg.supervision.rgb.loss_type,
            coef=cfg.supervision.rgb.loss_coef,
            name="rgb",
            check_nan=cfg.optim.check_nan,
        )

    if cfg.data.pixel_source.load_sky_mask and cfg.nerf.model.head.enable_sky_head:
        sky_loss_fn = loss.SkyLoss(
            loss_type=cfg.supervision.sky.loss_type,
            coef=cfg.supervision.sky.loss_coef,
            check_nan=cfg.optim.check_nan,
        )
    else:
        sky_loss_fn = None
    ## ------ dynamic related losses -------- #
    if cfg.nerf.model.head.enable_dynamic_branch:
        dynamic_reg_loss_fn = loss.DynamicRegularizationLoss(
            loss_type=cfg.supervision.dynamic.loss_type,
            coef=cfg.supervision.dynamic.loss_coef,
            entropy_skewness=cfg.supervision.dynamic.entropy_loss_skewness,
            check_nan=cfg.optim.check_nan,
        )
    else:
        dynamic_reg_loss_fn = None

    if cfg.nerf.model.head.enable_shadow_head:
        shadow_loss_fn = loss.DynamicRegularizationLoss(
            name="shadow",
            loss_type=cfg.supervision.shadow.loss_type,
            coef=cfg.supervision.shadow.loss_coef,
            check_nan=cfg.optim.check_nan,
        )
    else:
        shadow_loss_fn = None
    from dataLoader.waymo import WaymoDataset
    waymodataset = WaymoDataset(data_cfg=cfg.data, cfg_pose=cfg.pose)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = builders.build_model_from_cfg(
        cfg=cfg.nerf.model, dataset=waymodataset, device=device
    )
    optimizer = builders.build_optimizer_from_cfg(cfg=cfg.optim, model=model)
    (
        proposal_estimator,
        proposal_networks,
    ) = builders.build_estimator_and_propnet_from_cfg(
        nerf_cfg=cfg.nerf, optim_cfg=cfg.optim, dataset=waymodataset, device=device
    )
    model.train()
    proposal_estimator.train()
    for p in proposal_networks:
        p.train()
    proposal_requires_grad_fn = get_proposal_requires_grad_fn()
    step = 0
    pixel_loss_dict = {}
    while True:
        proposal_requires_grad = proposal_requires_grad_fn(int(step))
        i = torch.randint(0, len(waymodataset.train_pixel_set), (1,)).item()
        pixel_data_dict = waymodataset.train_pixel_set[i]
        for k, v in pixel_data_dict.items():
            if isinstance(v, torch.Tensor):
                pixel_data_dict[k] = v.cuda(non_blocking=True)
        # ------ pixel-wise supervision -------- #
        render_results = render_rays(
            radiance_field=model,
            proposal_estimator=proposal_estimator,
            proposal_networks=proposal_networks,
            data_dict=pixel_data_dict,
            cfg=cfg,
            proposal_requires_grad=proposal_requires_grad,
        )
        proposal_estimator.update_every_n_steps(
            render_results["extras"]["trans"],
            proposal_requires_grad,
            loss_scaler=1024,
        )

        # rgb loss
        pixel_loss_dict.update(
            rgb_loss_fn(render_results["rgb"], pixel_data_dict["pixels"])
        )
        
        # total_loss = 1.0 * ((torch.abs(render_results["rgb"] - pixel_data_dict["pixels"]))).mean()
        if dynamic_reg_loss_fn is not None:
                pixel_loss_dict.update(
                    dynamic_reg_loss_fn(
                        dynamic_density=render_results["extras"]["dynamic_density"],
                        static_density=render_results["extras"]["static_density"],
                    )
                )
        if shadow_loss_fn is not None:
                pixel_loss_dict.update(
                    shadow_loss_fn(
                        render_results["shadow_ratio"],
                    )
                )
        if sky_loss_fn is not None:  # if sky loss is enabled
            if cfg.supervision.sky.loss_type == "weights_based":
                # penalize the points' weights if they point to the sky
                pixel_loss_dict.update(
                    sky_loss_fn(
                        render_results["extras"]["weights"],
                        pixel_data_dict["sky_masks"],
                    )
                )
            elif cfg.supervision.sky.loss_type == "opacity_based":
                # penalize accumulated opacity if the ray points to the sky
                pixel_loss_dict.update(
                    sky_loss_fn(
                        render_results["opacity"], pixel_data_dict["sky_masks"]
                    )
                )
            else:
                raise NotImplementedError(
                    f"sky_loss_type {cfg.supervision.sky.loss_type} not implemented"
                )
        total_loss = sum(loss for loss in pixel_loss_dict.values())
        print("loss", total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        step += 1
        if step % 2000 == 0:
            # do_evaluation(
            #     step=step,
            #     cfg=cfg,
            #     model=model,
            #     proposal_networks=proposal_networks,
            #     proposal_estimator=proposal_estimator,
            #     dataset=waymodataset,
            #     args=args,
            # )
            model.eval()
            proposal_estimator.eval()
            for p in proposal_networks:
                p.eval()
            # waymodataset.pixel_source.update_downscale_factor()

            render_results = render_pixels(
                cfg=cfg,
                model=model,
                proposal_estimator=proposal_estimator,
                dataset=waymodataset.full_pixel_set,
                proposal_networks=proposal_networks,
                compute_metrics=True,
                return_decomposition=True,
                vis_indices=[waymodataset.test_pixel_set.split_indices[0]],
            )
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "feat_psnr",
                    "masked_psnr",
                    "masked_ssim",
                    "masked_feat_psnr",
                ]:
                    eval_dict[f"pixel_metrics/test/{k}"] = v
            print(f"test_psnr: {render_results['psnr']}\n")

            vis_frame_dict = save_videos(
                    render_results,
                    save_pth=os.path.join(
                        args.logdir, "images", f"step_{step}.png"
                    ),  # don't save the video
                    num_timestamps=1,
                    keys=render_keys,
                    save_seperate_video=cfg.logging.save_seperate_video,
                    num_cams=waymodataset.pixel_source.num_cams,
                    fps=cfg.render.fps,
                    verbose=False,
                )



        


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    cfg = setup(args)
    
    reconstruction(args)
