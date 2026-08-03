"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
Sep 10, 2020
"""
import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
import glob

class arbitrary(AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, tamp_list: str, read_from_jpeg=False, dic_cover=None):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list:
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self.tamp_list = tamp_list
        self.cover_dic = dic_cover
        self.read_from_jpeg = read_from_jpeg

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self.tamp_list[index]
        im = Image.open(tamp_path)
        if im.format != "JPEG":
            temp_jpg = f"____temp_{index:04d}.jpg"
            Image.open(tamp_path).convert('RGB').save(temp_jpg, quality=100, subsampling=0)
            image, label, qtable = self._create_tensor(temp_jpg, None)
            os.remove(temp_jpg)
        else:
            image, label, qtable = self._create_tensor(tamp_path, None)
        if self.cover_dic and tamp_path in self.cover_dic and self.cover_dic[tamp_path] is not None:
            cover_path = self.cover_dic[tamp_path]
            cover_L = np.array(Image.open(cover_path).convert("L"))
        else:
            cover_L = np.zeros(im.size[::-1], dtype=np.uint8)
        return (
            image, label, qtable, 
            torch.tensor(cover_L, dtype=torch.float).unsqueeze(0) / 255.0
        )
