"""
Created by Myung-Joon Kwon
mjkwon2021@gmail.com
July 8, 2020
"""


import torch
from torch.utils.data import Dataset
import random
from Splicing.data.dataset_arbitrary import arbitrary


class SplicingDataset(Dataset):
    def __init__(self, crop_size, grid_crop, blocks=('RGB',), mode="train", DCT_channels=3, read_from_jpeg=False, class_weight=None, list_img=None, dic_cover=None):
        self.dataset = arbitrary(crop_size, grid_crop, blocks, DCT_channels, list_img, read_from_jpeg=read_from_jpeg, dic_cover=dic_cover)
        if class_weight is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weight)
        self.crop_size = crop_size
        self.grid_crop = grid_crop
        self.blocks = blocks
        self.mode = mode
        self.read_from_jpeg = read_from_jpeg
        self.smallest = 1869  # smallest dataset size (IMD:1869)

    def shuffle(self):
        random.shuffle(self.dataset.tamp_list)

    def get_PIL_image(self, index):
        assert self.dataset.tamp_list
        assert 0 <= index < len(self.dataset.tamp_list), f"Index {index} is not available!"
        return self.dataset.get_PIL_Image(index)

    def get_filename(self, index):
        assert self.dataset.tamp_list
        assert 0 <= index < len(self.dataset.tamp_list), f"Index {index} is not available!"
        return self.dataset.get_tamp_name(index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        assert self.dataset.tamp_list
        assert 0 <= index < len(self.dataset.tamp_list), f"Index {index} is not available!"
        return self.dataset.get_tamp(index)

    def get_info(self):
        s = ""
        s += (str(self.dataset)+'('+str(len(self.dataset))+') ')
        s += '\n'
        s += f"crop_size={self.crop_size}, grid_crop={self.grid_crop}, blocks={self.blocks}, mode={self.mode}, read_from_jpeg={self.read_from_jpeg}, class_weight={self.class_weights}\n"
        return s





