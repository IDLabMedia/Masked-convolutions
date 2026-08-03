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
"""
"""
Edited in March 2026
@author: xander.staelens
"""

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from matplotlib import colormaps
from PIL import Image

import torch
from torch.nn import functional as F

from TruFor_train_test.lib.config import config, update_config
from TruFor_train_test.lib.utils import get_model
from TruFor_train_test.dataset.dataset_test import TestDataset

parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-g',   '--gpu',     type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-w',   '--weights', type=str, default='../weights/trufor.pth.tar', help='path to the model weights')

parser.add_argument('-in',  '--input',   type=str, default='../images', help='can be a single file, a directory or a glob statement')
parser.add_argument('-cover',  '--cover',   type=str, default='../covers', help='can be a single file (if input is a single file, can be different name), or a directory (if input is directory or glob, and same file names as input) [only PNG, same size as input]')
parser.add_argument('-out', '--output',  type=str, default='../output', help='output folder')

parser.add_argument('-idc', '--iterative-distraction-cover', action='store_true', help='auto generate distraction cover iteratively during inference')
parser.add_argument('-irn', '--iterative-round-number', type=int, default=2, help='number of rounds for iterative distraction cover generation (1 means no distraction cover is used)')
parser.add_argument('-itr', '--iterative-threshold', type=float, default=0.5, help='threshold for iterative distraction cover generation')
parser.add_argument('-idl', '--iterative-dilation', type=float, default=5.0, help='dilation (in % of image width) for iterative distraction cover generation')

parser.add_argument('-exp', '--experiment', type=str, default='trufor_ph3')

parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input   = args.input
cover   = args.cover
output  = args.output
gpu     = args.gpu

iterative_distraction_cover = args.iterative_distraction_cover
iterative_round_number = args.iterative_round_number
iterative_threshold = args.iterative_threshold
iterative_dilation = args.iterative_dilation

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

if '*' in input:
    list_img = glob(input, recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
    dic_cover = {img: os.path.join(cover, os.path.basename(os.path.splitext(img)[0]) + '.png') for img in list_img}
    dic_cover = {img: cover if os.path.isfile(cover) else None for img, cover in dic_cover.items()}
elif os.path.isfile(input):
    list_img = [input]
    dic_cover = {input: cover if os.path.isfile(cover) else None}
elif os.path.isdir(input):
    list_img = glob(os.path.join(input, '**/*'), recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
    dic_cover = {img: os.path.join(cover, os.path.basename(os.path.splitext(img)[0]) + '.png') for img in list_img}
    dic_cover = {img: cover if os.path.isfile(cover) else None for img, cover in dic_cover.items()}
else:
    raise ValueError("input is neither a file or a folder")

print(f"Images found: {list_img}")
print(f"Covers found: {dic_cover}")

test_dataset = TestDataset(list_img=list_img, dic_cover=dic_cover)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1)   # 1 to allow arbitrary input sizes

if args.weights:
    model_state_file = args.weights
else:
    raise ValueError("Model file is not specified.")

print('=> loading model from {}'.format(model_state_file))
checkpoint = torch.load(model_state_file, map_location=torch.device(device), weights_only=False)
print("Epoch: {}".format(checkpoint['epoch']))

model = get_model(config)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

with torch.no_grad():
    for index, (rgb, cover, path) in enumerate(tqdm(testloader)):

        ## --- Path ---
        if os.path.splitext(os.path.basename(output))[1] == '':  # output is a directory
            path = path[0]
            root = input.split('*')[0]

            if os.path.isfile(input):
                sub_path = path.replace(os.path.dirname(root), '').strip()
            else:
                sub_path = path.replace(root, '').strip()

            if sub_path.startswith('/'):
                sub_path = sub_path[1:]

            filename = "TruFor_" + sub_path

            # if anything is covered
            if cover is not None and torch.sum(cover) > 0:
                filename += "_masked"

            if iterative_distraction_cover:
                filename += "_iterative_{:02d}_th_{:.2f}_dil_{:.2f}".format(iterative_round_number, iterative_threshold, iterative_dilation)

            filename_out = os.path.join(output, filename) + '.npz'
        else:  # output is a filename
            filename_out = output

        if not filename_out.endswith('.npz'):
            filename_out = filename_out + '.npz'


        ## --- Sample ---

        try:
            rgb = rgb.to(device)
            cover = cover.to(device)
            model.eval()

            det  = None
            conf = None

            B, C, H, W = rgb.shape


            ## --- Default, using given distraction cover ---

            if not iterative_distraction_cover:
                pred, *_ = model(rgb, distraction_cover=cover)
                pred = torch.squeeze(pred, 0)
                pred = F.softmax(pred, dim=0)[1]
                pred = pred.cpu().numpy()

            ## --- Iteratively generate and use distraction covers ---

            else:
                merged_pred = None
                current_cover = None

                for round_idx in range(iterative_round_number):
                    pred, *_ = model(rgb, distraction_cover=current_cover)
                    pred = torch.squeeze(pred, 0)
                    pred = F.softmax(pred, dim=0)[1]
                    pred = pred.cpu().numpy()

                    if merged_pred is None:
                        merged_pred = pred.copy() # first round
                    else:
                        unmasked_indices = (current_cover[0,0,:,:] < 0.5).cpu().numpy() # batch size 1
                        merged_pred[unmasked_indices] = pred[unmasked_indices]

                    # Generate new distraction cover based on merged prediction
                    current_cover = torch.zeros((1,1,H,W), dtype=torch.float32, device=device)
                    current_cover[0,0,:,:] = torch.tensor(merged_pred, device=device) >= iterative_threshold

                    # Perform "binary dilation" using max pooling via convolution
                    dilation_pixels = int(iterative_dilation/100 * max(W, H))
                    if dilation_pixels % 2 == 0:
                        dilation_pixels += 1  # make sure it's odd
                    kernel = torch.ones((1, 1, dilation_pixels, dilation_pixels), device=device)
                    dilated_cover = F.conv2d(current_cover, kernel, padding=dilation_pixels // 2)
                    # Any positive value in the window => dilated pixel = 1
                    dilated_cover = (dilated_cover > 0).float()

                    # Put result back
                    current_cover[0, 0, :, :] = dilated_cover[0, 0]

                pred = merged_pred


            out_dict = dict()
            out_dict['map'    ] = pred
            out_dict['imgsize'] = tuple(rgb.shape[2:])

            from os import makedirs
            makedirs(os.path.dirname(filename_out), exist_ok=True)
            np.savez(filename_out, **out_dict)

            # also save output as image
            filename_out_img = filename_out[:-4] + '.png'
            cm_ = colormaps.get_cmap('RdBu_r')
            jet = (cm_(pred)[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(jet).save(filename_out_img)


        except:
            import traceback
            traceback.print_exc()
            pass

