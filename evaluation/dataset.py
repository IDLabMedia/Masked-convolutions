"""
Dataset for the forensic-distraction evaluation splits (evaluation/splits/*.csv).

Each split CSV describes, per image, which distraction (from evaluation/distractions/)
to paste on top of it and where. This file loads the base image/mask, composites the
distraction, and returns everything needed to evaluate (or visualize) an IFL model:

    label                 0 (authentic) / 1 (forged)
    image, mask           paths relative to --dataset-folder ("none" mask if label==0)
    distraction           filename in --distraction-folder, or "none"
    distraction_size      distraction width, as a % of the composited image's long side
    distraction_position  "(x%, y%)" top-left position, as a % of the free space it can occupy
    quality               JPEG quality applied after compositing
    width                 (optional) fixed long-side size images/masks are resized to
    invert_mask           (optional) invert the loaded ground-truth mask (needed for DSO-1)

Run this file directly to see examples of a split, e.g.:

    python dataset.py splits/DSO-1_random_distractions_QF_(85-95)_small_(4.0-12.0).csv \
        -d ./datasets -dis ./distractions -out ./output
"""

import argparse
import ast
import io
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# Fixed aspect ratio (width/height) the splits were generated with, e.g. 1536x1024.
TARGET_ASPECT = 768 / 512


def resize_and_crop(image, long_side):
    """Resize + center-crop `image` to `long_side` x `long_side / TARGET_ASPECT`
    (or the portrait equivalent), matching how the splits were built."""
    short_side = round(long_side / TARGET_ASPECT)
    size = (short_side, long_side) if image.width < image.height else (long_side, short_side)
    return ImageOps.fit(image, size, method=Image.LANCZOS, centering=(0.5, 0.5))


def to_tensor(array, dtype=torch.float32, scale=1.0):
    """(H, W) or (H, W, C) numpy array -> (C, H, W) tensor."""
    return torch.tensor(np.atleast_3d(array).transpose(2, 0, 1), dtype=dtype) * scale


class DistractionDataset(Dataset):
    """Loads a base image/mask and composites the distraction described by each CSV row."""

    def __init__(self, split_csv, dataset_folder, distraction_folder):
        self.rows = pd.read_csv(split_csv, delimiter=";")
        self.dataset_folder = dataset_folder
        self.distraction_folder = distraction_folder

    def __len__(self):
        return len(self.rows)

    def build(self, index):
        """Composite row `index` and return plain PIL/numpy objects (no torch):
        (row, image (RGB, already JPEG round-tripped), mask (L), distraction_mask,
        distraction_cover). The last two are (H, W) uint8 arrays, or None if the
        row has no distraction ("distraction" column missing/"none")."""
        row = self.rows.iloc[index]
        label = int(row["label"])
        long_side = int(row["width"]) if row.get("width", 0) > 0 else 0

        image = Image.open(os.path.join(self.dataset_folder, row["image"])).convert("RGBA")
        if long_side:
            image = resize_and_crop(image, long_side)

        if label:
            mask = Image.open(os.path.join(self.dataset_folder, row["mask"])).convert("L")
            if bool(row.get("invert_mask", False)):
                mask = ImageOps.invert(mask)
            if long_side:
                mask = resize_and_crop(mask, long_side)
        else:
            mask = Image.new("L", image.size, 0)

        distraction_mask = distraction_cover = None
        if row.get("distraction", "none") != "none":
            distraction_mask, distraction_cover = self._paste_distraction(image, mask, row, label)

        # JPEG-compress in memory (no temp files on disk)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=int(row.get("quality", 100)))
        image = Image.open(buffer).convert("RGB")

        return row, image, mask, distraction_mask, distraction_cover

    def __getitem__(self, index):
        row, image, mask, distraction_mask, distraction_cover = self.build(index)
        empty = torch.zeros(1, image.height, image.width)
        return {
            "label": int(row["label"]),
            "image_name": row["image"],
            "distracted": distraction_mask is not None,
            "image": to_tensor(np.array(image), scale=1 / 255),
            "mask": to_tensor(np.array(mask), dtype=torch.long),
            "distraction_mask": to_tensor(distraction_mask, scale=1 / 255) if distraction_mask is not None else empty,
            "distraction_cover": to_tensor(distraction_cover, scale=1 / 255) if distraction_cover is not None else empty,
        }

    def _paste_distraction(self, image, mask, row, label):
        """Paste the distraction onto `image` (in place), carve it out of `mask` (in
        place), and return (distraction_mask, distraction_cover) as (H, W) uint8 arrays
        (255 = distraction). `distraction_mask` follows the distraction's alpha shape;
        `distraction_cover` is its full bounding box (used to mask convolutions)."""
        distraction = Image.open(os.path.join(self.distraction_folder, row["distraction"])).convert("RGBA")

        width = int(max(image.size) * float(row["distraction_size"]) / 100)
        height = int(width * distraction.height / distraction.width)
        distraction = distraction.resize((width, height), Image.LANCZOS)

        pos_x_pct, pos_y_pct = ast.literal_eval(row["distraction_position"])
        max_x, max_y = max(image.width - width, 0), max(image.height - height, 0)
        x, y = int(pos_x_pct * max_x / 100), int(pos_y_pct * max_y / 100)
        box = (x, y, x + width, y + height)

        image.paste(distraction, box, distraction)
        if label:
            mask.paste(Image.new("L", distraction.size, 0), box, distraction)

        distraction_mask = Image.new("L", image.size, 0)
        distraction_mask.paste(Image.new("L", distraction.size, 255), box, distraction)

        distraction_cover = Image.new("L", image.size, 0)
        distraction_cover.paste(Image.new("L", distraction.size, 255), box)

        return np.array(distraction_mask), np.array(distraction_cover)


def build_distracted_dataset(split_csv, dataset_folder, distraction_folder, out_folder):
    """Materialize every row of `split_csv` to disk as: <name>.jpg (distracted image),
    <name>_mask.png (ground-truth mask) and <name>_cover.png (distraction cover, if any)."""
    dataset = DistractionDataset(split_csv, dataset_folder, distraction_folder)
    os.makedirs(out_folder, exist_ok=True)

    for i in range(len(dataset)):
        row, image, mask, _distraction_mask, distraction_cover = dataset.build(i)
        name = os.path.splitext(os.path.basename(row["image"]))[0]

        image.save(os.path.join(out_folder, f"{name}.jpg"), quality=100)
        mask.save(os.path.join(out_folder, f"{name}_mask.png"))
        if distraction_cover is not None:
            Image.fromarray(distraction_cover).save(os.path.join(out_folder, f"{name}_cover.png"))

        print(f"[{i + 1}/{len(dataset)}] saved {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_csv", help="path to a split CSV, e.g. splits/DSO-1_..._small_....csv")
    parser.add_argument("-d", "--dataset-folder", default="./images", help="folder with the base images/masks")
    parser.add_argument("-dis", "--distraction-folder", default="./distractions", help="folder with the distraction PNGs")
    parser.add_argument("-out", "--out-folder", default="./output/distracted", help="where to save the distracted images")
    args = parser.parse_args()

    build_distracted_dataset(args.split_csv, args.dataset_folder, args.distraction_folder, args.out_folder)
