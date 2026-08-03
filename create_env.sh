#!/bin/bash

mkdir -p env
mkdir -p env/conda/envs
mkdir -p env/conda/pkgs

conda config --add envs_dirs env/conda/envs
conda config --add pkgs_dirs env/conda/pkgs

conda config --show envs_dirs
conda config --show pkgs_dirs

if [ ! -f "env/forensic/bin/python" ]; then
    echo "Creating environment..."
    conda create --prefix env/forensic python=3.8
    conda activate env/forensic
else
    echo "Environment already exists."
fi

echo "Updating packages..."

conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia

env/forensic/bin/pip install \
    opencv-python \
    pillow \
    pyyaml \
    tqdm \
    yacs \
    torch-dct \
    jpegio \
    seaborn \
    albumentations \
    imageio \
    tensorboardX \
    timm \
    matplotlib