from data.img_utils import (
    normalize_rgb_img,
    read_rgba_img,
)

import glob
import os
from torch.utils.data import Dataset
import torch
import numpy as np


class CorruptedImagesDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.src_image_files = sorted(
            glob.glob(os.path.join(root_dir, "src_imgs/*.png"))
        )
        self.corrupted_image_files = sorted(
            glob.glob(os.path.join(root_dir, "corrupted_imgs/*.png"))
        )
        self.mask_files = sorted(
            glob.glob(os.path.join(root_dir, "binary_masks/*.npy"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.src_image_files)

    def __getitem__(self, idx):
        # print(self.corrupted_image_files[idx])
        # print(self.mask_files[idx])
        # print(self.src_image_files[idx])

        corrupted_img = read_rgba_img(img_path=self.corrupted_image_files[idx])
        norm_corrupted_img = torch.from_numpy(normalize_rgb_img(img=corrupted_img))

        binary_mask = torch.from_numpy(np.load(self.mask_files[idx]))

        src_img = read_rgba_img(img_path=self.src_image_files[idx])
        norm_src_img = torch.from_numpy(normalize_rgb_img(img=src_img))

        return norm_corrupted_img, binary_mask, norm_src_img
