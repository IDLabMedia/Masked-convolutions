# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro

Edited in September 2025
@author: xander.staelens
"""

from torch.utils.data import Dataset
import random
import numpy as np
import torch
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, list_img=None, dic_cover=None):
        self.img_list = list_img
        self.cover_dic = dic_cover

    def shuffle(self):
        random.shuffle(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        assert self.img_list
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        rgb_path = self.img_list[index]
        img_RGB = np.array(Image.open(rgb_path).convert("RGB"))
        if self.cover_dic and rgb_path in self.cover_dic and self.cover_dic[rgb_path] is not None:
            cover_path = self.cover_dic[rgb_path]
            cover_L = np.array(Image.open(cover_path).convert("L"))
        else:
            cover_L = np.zeros(img_RGB.shape[:2], dtype=np.uint8)
        return (
            torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, 
            torch.tensor(cover_L, dtype=torch.float).unsqueeze(0) / 255.0,
            rgb_path
        )

    def get_filename(self, index):
        item = self.img_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item