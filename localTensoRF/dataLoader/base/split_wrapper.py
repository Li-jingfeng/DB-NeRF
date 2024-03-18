from typing import List, Union

import torch
import torch.nn.functional as F

from .lidar_source import SceneLidarSource
from .pixel_source import ScenePixelSource
import numpy as np


class SplitWrapper(torch.utils.data.Dataset):

    # a sufficiently large number to make sure we don't run out of data
    _num_iters = 1000000

    def __init__(
        self,
        datasource: Union[ScenePixelSource, SceneLidarSource],
        split_indices: List[int] = None,
        split: str = "train",
        ray_batch_size: int = 4096,
        num_frame_per_nerf: int = None, #每个nerf有多少个frames
        test_image_start_id: int = None, #第一个nerf测试起始帧
    ):
        super().__init__()
        self.datasource = datasource
        self.split_indices = split_indices
        self.num_images = len(split_indices) #所有图像
        self.split = split
        self.ray_batch_size = ray_batch_size
        self.train_cam_id_count = -1
        
        self.num_frame_per_nerf = num_frame_per_nerf
        
        self.test_frame_idx = 5
        self.vis_frames = [0, 1, 2, 3, 4, 6, 7, 8, 9]
        
    def __getitem__(self, idx) -> dict:
        if isinstance(idx, tuple):
            model_index = idx[1]
            idx = idx[0]
        if self.split == "train":
            # randomly sample rays from the training set
            # self.train_cam_id_count = (self.train_cam_id_count + 1) % len(self.split_indices)
            return self.datasource.get_train_rays(
                num_rays=self.ray_batch_size,
                candidate_indices=self.vis_frames,
                # candidate_indices=self.train_cam_id_count,
            )
        elif self.split == "test":
            return self.datasource.get_train_rays(
                num_rays=self.ray_batch_size,
                # candidate_indices=self.test_frame_idx,
                # candidate_indices=9,
                candidate_indices=self.test_frame_idx,
                # candidate_indices=self.train_cam_id_count,
            )
        else:
            # return all rays for the given index
            return self.datasource.get_render_rays(self.split_indices[idx])

    # 每当切换model的时候更新要训练的frames
    def update_train_frames(self, model_index):
        self.vis_frames = model_index

    # test idx per model
    def update_test_frames(self, model_index):
        self.test_frame_idx = model_index
        
    def __len__(self) -> int:
        if self.split == "train":
            return self.num_iters
        else:
            return len(self.split_indices)

    @property
    def num_iters(self) -> int:
        return self._num_iters

    def set_num_iters(self, num_iters) -> None:
        self._num_iters = num_iters
