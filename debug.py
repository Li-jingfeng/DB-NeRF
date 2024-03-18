import torch.nn.functional as F
import numpy as np
import imageio
import cv2
from third_party import pytorch_ssim
import torch

def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

def compute_metric(pred_image_path, gt_image_path):
    img_pred = imread(pred_image_path) / 255
    img_gt = imread(gt_image_path) / 255
    img_pred = torch.from_numpy(img_pred)
    img_gt = torch.from_numpy(img_gt)
    # mse for the entire image
    mse = F.mse_loss(img_pred, img_gt).item()
    psnr = mse2psnr(mse)
    ssim = pytorch_ssim.ssim(img_pred.permute(2, 0, 1).unsqueeze(0), img_gt.permute(2, 0, 1).unsqueeze(0)).item()
    print(f"psnr:{psnr}, ssim:{ssim}")
    return psnr, ssim

def imread(f):
    rgb = imageio.imread(f)
    if rgb.shape[:2] != (640, 960): 
    # rgb_resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.resize(
                rgb, (960, 640), interpolation=cv2.INTER_LINEAR
        )
    return rgb

import matplotlib.pyplot as plt
localrf = "/data/ljf/localrf/log/original_resolution/rgb_maps/020_0.jpg"
ours = "/data/ljf/localrf/log/recover/rgb_maps/020_0.jpg"
gt_image_path = "/data/ljf/localrf/waymo_dynamic/waymo/processed/training/016/images/020_0.jpg"

local_metric = compute_metric(localrf, gt_image_path)
ours_metric = compute_metric(ours, gt_image_path)
print(f"localrf:{local_metric}, ours:{ours_metric}")