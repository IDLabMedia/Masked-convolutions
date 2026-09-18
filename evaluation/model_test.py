"""
Evaluate TruFor / CAT-Net with or without masked convolutions on a CSV split,
composited on the fly with dataset.DistractionDataset, reporting IoU / MCC / F1
per image.

Model (--model):
    TruFor | TruFor_masked | CAT-Net | CAT-Net_masked
    The "_masked" variants pass a distraction cover to the model; the plain
    variants never do. Both use the exact same (unmodified) weights -- masked
    convolutions need no retraining, so "masked" purely means "a cover is
    passed at inference time".

Masking (--masking, only affects the "_masked" variants):
    none        never pass a cover (behaves like the plain variant)
    auto        use the exact distraction cover the dataset composited with
                (a "perfect"/oracle mask)
    iterative   auto-generate the cover in 2 rounds: round 1 runs without a
                cover, its prediction is thresholded at 0.5 and dilated by 5%
                of the image's longest side to build the cover for round 2

Works with both the distraction splits (evaluation/splits/*.csv) and plain,
distraction-free CSVs (label;image;mask[;quality][;width][;invert_mask])

Example (run from the evaluation/ folder):
    python model_test.py -m TruFor_masked -mask iterative \
        -csv "splits/IMD2020_random_distractions_QF_(85-95)_small_(4.0-12.0).csv" \
        -d ./datasets -dis ./distractions -w ../weights/trufor.pth.tar \
        -out ./output/results_IMD2020_small_iterative.csv
"""

import argparse
import contextlib
import importlib
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataset import DistractionDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUFOR_DIR = os.path.join(REPO_ROOT, "TruFor")
CATNET_DIR = os.path.join(REPO_ROOT, "CAT-Net")

# Iterative-masking settings
ITERATIVE_ROUNDS = 2           # round 1 = no cover, round 2 = generated cover
ITERATIVE_THRESHOLD = 0.5
ITERATIVE_DILATION_PCT = 5.0   # % of the image's longest side


@contextlib.contextmanager
def _use_repo(folder):
    """TruFor's and CAT-Net's own code load config/weights via paths relative
    to their own repo root, so temporarily cd + sys.path into it."""
    old_cwd, old_path = os.getcwd(), list(sys.path)
    os.chdir(folder)
    sys.path.insert(0, folder)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


# --------------------------------------------------------------------------- #
# Model loading (mirrors each repo's own test.py)
# --------------------------------------------------------------------------- #

def load_trufor(weights, device, experiment="trufor_ph3"):
    with _use_repo(TRUFOR_DIR):
        from TruFor_train_test.lib.config import config, update_config
        from TruFor_train_test.lib.utils import get_model

        update_config(config, argparse.Namespace(experiment=experiment, gpu=[], opts=[]))
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        model = get_model(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()


def load_catnet(weights, device, experiment="CAT_full"):
    with _use_repo(CATNET_DIR):
        from lib.config import config, update_config

        update_config(config, argparse.Namespace(
            cfg=f"experiments/{experiment}.yaml",
            opts=["TEST.MODEL_FILE", weights, "TEST.FLIP_TEST", "False", "TEST.NUM_SAMPLES", "0"],
        ))
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        model = importlib.import_module(f"lib.models.{config.MODEL.NAME}").get_seg_model(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Per-architecture input preparation + inference (both take/return plain
# numpy/torch, so the masking logic below can stay architecture-agnostic)
# --------------------------------------------------------------------------- #

def trufor_infer_fn(model, image, device):
    """`image`: PIL RGB. Returns (infer_fn(cover) -> (H, W) float32 pred, shape)."""
    rgb = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device) / 255.0

    def infer(cover):
        pred, *_ = model(rgb, distraction_cover=cover)
        return F.softmax(torch.squeeze(pred, 0), dim=0)[1].detach().cpu().numpy()

    return infer, tuple(rgb.shape[2:])


def catnet_infer_fn(model, jpeg_path, device):
    """`jpeg_path`: path to the (already composited) JPEG on disk -- CAT-Net reads
    the JPEG's own DCT coefficients/quantization table, so it needs a real file."""
    with _use_repo(CATNET_DIR):
        from Splicing.data.dataset_test import SplicingDataset

        ds = SplicingDataset(crop_size=None, grid_crop=True, blocks=("RGB", "DCTvol", "qtable"),
                              DCT_channels=1, read_from_jpeg=True, list_img=[jpeg_path], dic_cover={jpeg_path: None})
        image, _label, qtable, _cover = ds[0]

    image = image.unsqueeze(0).to(device)
    qtable = qtable.unsqueeze(0).to(device)
    height, width = image.shape[2:]

    def infer(cover):
        pred = model(image, qtable, distraction_cover=cover)
        pred = F.softmax(torch.squeeze(pred, 0), dim=0)[1].detach().cpu().numpy()
        return upscale_mask(pred, height, width)

    return infer, (height, width)


def upscale_mask(mask, height, width):
    import cv2
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
# Masking modes (architecture-agnostic: just needs an `infer(cover)` closure)
# --------------------------------------------------------------------------- #

def run_masking(infer, mode, shape, device, gt_cover=None):
    """Returns a (H, W) float32 prediction in [0, 1] using the requested masking mode."""
    height, width = shape

    if mode == "none":
        return infer(None)

    if mode == "auto":
        cover = None
        if gt_cover is not None:
            cover = torch.tensor(gt_cover, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
        return infer(cover)

    if mode == "iterative":
        merged, cover = None, None
        for _round in range(ITERATIVE_ROUNDS):
            pred = infer(cover)
            if merged is None:
                merged = pred.copy()
            else:
                unmasked = cover[0, 0].cpu().numpy() < 0.5
                merged[unmasked] = pred[unmasked]

            # Threshold the merged prediction to build the next round's cover.
            cover = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
            cover[0, 0] = torch.tensor(merged, device=device) >= ITERATIVE_THRESHOLD

            # Dilate it (binary dilation via max-pool-like convolution).
            dilation_px = int(ITERATIVE_DILATION_PCT / 100 * max(width, height))
            dilation_px += 1 - dilation_px % 2  # make sure it's odd
            kernel = torch.ones((1, 1, dilation_px, dilation_px), device=device)
            dilated = F.conv2d(cover, kernel, padding=dilation_px // 2)
            cover[0, 0] = (dilated[0, 0] > 0).float()

        return merged

    raise ValueError(f"Unknown masking mode: {mode}")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def compute_scores(gt, pred):
    gt = (gt > 0.5).astype(np.float32)
    tp = np.sum(gt * pred)
    fn = np.sum(gt * (1 - pred))
    fp = np.sum((1 - gt) * pred)
    tn = np.sum((1 - gt) * (1 - pred))

    iou = tp / (tp + fn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)

    # Edge cases for Matthews correlation coefficient (MCC) calculation
    if tp == 0 and fp == 0:
        mcc = 1 if fn == 0 and tn != 0 else 0 if fn != 0 and tn != 0 else -1
    elif tp == 0 and fn == 0:
        mcc = 1 if fp == 0 and tn != 0 else 0 if fp != 0 and tn != 0 else -1
    elif tn == 0 and fp == 0:
        mcc = 1 if tp != 0 and fn == 0 else 0 if tp != 0 and fn != 0 else -1
    elif tn == 0 and fn == 0:
        mcc = 1 if tp != 0 and fp == 0 else 0 if tp != 0 and fp != 0 else -1
    else:
        # Standard MCC calculation for non-edge cases
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return iou, mcc, f1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    architecture, masked = ("TruFor", True) if args.model == "TruFor_masked" else \
                           ("TruFor", False) if args.model == "TruFor" else \
                           ("CAT-Net", True) if args.model == "CAT-Net_masked" else \
                           ("CAT-Net", False)
    masking = args.masking if masked else "none"
    if not masked and args.masking != "none":
        print(f"[model_test] Note: '{args.model}' is not a masked variant, ignoring --masking {args.masking!r}")

    print(f"[model_test] Loading {architecture} weights from {args.weights} ({device})")
    assert os.path.isfile(args.weights), f"Weights file not found: {args.weights}"
    if architecture == "TruFor":
        model = load_trufor(args.weights, device)
    else:
        model = load_catnet(args.weights, device)

    dataset = DistractionDataset(args.csv, args.dataset_folder, args.distraction_folder)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w") as f:
        f.write("label;image;distracted;IoU;MCC;F1\n")

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            row, image, mask, distraction_mask, distraction_cover = dataset.build(i)

            if architecture == "TruFor":
                infer, shape = trufor_infer_fn(model, image, device)
            else:
                # CAT-Net needs the composited image as an actual JPEG file on disk.
                jpeg_path = os.path.join(args.tmp_folder, f"__catnet_tmp_{i}.jpg")
                os.makedirs(args.tmp_folder, exist_ok=True)
                image.save(jpeg_path, format="JPEG", quality=100)
                try:
                    infer, shape = catnet_infer_fn(model, jpeg_path, device)
                finally:
                    os.remove(jpeg_path)

            pred = run_masking(infer, masking, shape, device, gt_cover=distraction_cover)

            gt = (np.array(mask) > 0).astype(np.float32)
            if pred.shape != gt.shape:  # e.g. CAT-Net's internal grid-cropping
                pred = F.interpolate(torch.tensor(pred).view(1, 1, *pred.shape), size=gt.shape,
                                      mode="bilinear", align_corners=False).view(gt.shape).numpy()

            valid = None
            if distraction_mask is not None:
                distraction_pixels = distraction_mask > 127
                if masking == "auto" and distraction_cover is not None:
                    valid = distraction_cover <= 127
                elif masking == "iterative":
                    valid = ~distraction_pixels
                else:
                    distraction_alpha = distraction_mask.astype(np.float32) / 255.0
                    gt = np.clip(gt - distraction_alpha, a_min=0, a_max=None)
                    pred = np.clip(pred - distraction_alpha, a_min=0, a_max=None)

            if valid is not None:
                gt, pred = gt[valid], pred[valid]
            iou, mcc, f1 = compute_scores(gt, pred)

            with open(args.out, "a") as f:
                f.write(f"{row['label']};{row['image']};{distraction_mask is not None};{iou};{mcc};{f1}\n")

    print(f"[model_test] Saved scores to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, choices=["TruFor", "TruFor_masked", "CAT-Net", "CAT-Net_masked"])
    parser.add_argument("-mask", "--masking", default="none", choices=["none", "auto", "iterative"],
                         help="only used by the '_masked' variants")
    parser.add_argument("-csv", "--csv", required=True, help="path to a CSV split (with or without distractions)")
    parser.add_argument("-d", "--dataset-folder", default="./datasets", help="folder with the base images/masks")
    parser.add_argument("-dis", "--distraction-folder", default="./distractions", help="folder with the distraction PNGs")
    parser.add_argument("-w", "--weights", required=True, help="path to the model's .pth/.pth.tar weights")
    parser.add_argument("-out", "--out", default="./output/results.csv", help="where to write per-image scores")
    parser.add_argument("-tmp", "--tmp-folder", default="./_tmp", help="scratch folder for CAT-Net temp JPEGs")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, help="device, use -1 for cpu")
    main(parser.parse_args())
