"""
This script is dedicated to the segmentation of cell membranes in Zebrafish embryos acquired in 3D with a spinning disk
microscope.

Author: Clément Cazorla
Date: 05/15/2024
Version: 1.0

It segments slice by slice in all 3 directions using Cellpose 3.0 (or any 2D cell segmentation model) and recombines the
segmentation masks using Cellstitch to get the final 3D mask.
"""


import os
import time
import argparse

import numpy as np
import torch
import tifffile
from cellpose.models import Cellpose
from skimage import io

from cellstitch.pipeline import full_stitch
# %%
# Plotting specifications
from matplotlib import rcParams
#from IPython.display import display

rcParams.update({'font.size': 10})
# %% md

# Definition of input parameters

parser = argparse.ArgumentParser(description='Parameters to set to segment zebrafish 3D embryo membranes.')
parser.add_argument('-f', type=str, help='Folder path containing the images to segment')
parser.add_argument('-ft', type=float, help='Flow threshold for Cellpose', default=0.4)
parser.add_argument('-mt', type=str, help='Cellpose model', default="cyto3")
parser.add_argument('-xyd', type=int, help='xy diameter for Cellpose Segmentation', default=250)
parser.add_argument('-zd', type=int, help='xz and yz diameter for Cellpose Segmentation', default=200)
parser.add_argument('-c', type=str, help='Channel to segment (wavelength)', default=561)

args = parser.parse_args()

folder = args.f
xy_diameter = args.xyd
z_diameter = args.zd
channel = args.c
flow_threshold = args.ft
model_type = args.mt

# Liste pour stocker les noms des fichiers filtrés
images_path = []

# Parcourir les fichiers dans le dossier
for fichier in os.listdir(folder):
    if fichier.endswith('.TIF') and channel + ' Em' in fichier:
        images_path.append(fichier)

print("Images to be processed:", images_path)

# Fill in on the path you would like to store the stitched mask
output_path = os.path.join(folder, "segmentation_masks")

if os.path.exists(output_path) is False:
    os.mkdir(output_path)


# %% md
# (1). Define configs & parameters
# %%
# load cellpose model for backbone segmentation
# you can also replace with any 2D segmentation model that works the best for your dataset
flow_threshold = flow_threshold
use_gpu = True if torch.cuda.is_available() else False
if use_gpu is True:
    print("Using GPU")

model = Cellpose(model_type='cyto3', gpu=use_gpu)
print("Cellpose model {} loaded".format(model_type))

# (2). Load example pairs of raw image & ground-truth mask
# %%
# Process all the images of the folder
tic = time.time()
for i, filename in enumerate(images_path):
    output_filename = os.path.splitext(filename)[0] + "_mask" + ".tif"

    # Load image & masks
    if filename[-3:] == 'npy':  # image in .npy format
        img = np.load(os.path.join(folder, filename))
    elif filename[-3:] == 'tif':  # imagge in TIFF format
        img = tifffile.imread(os.path.join(folder, filename))
    else:
        try:
            img = io.imread(os.path.join(folder, filename))
        except:
            raise IOError('Failed to load image {}'.format(filename))

    # %% md
    # (3). Run CellStitch
    # %%

    print(f"Processing image {i + 1} over {len(images_path)}: {filename}")

    print("computing xy masks")
    xy_masks = model.eval(list(img), flow_threshold=flow_threshold, channels=[0, 0], diameter=xy_diameter)[0]
    xy_masks = np.array(xy_masks)
    print("xy masks computed")

    print("computing yz masks")
    yz_masks = model.eval(list(img.transpose(1, 0, 2)), flow_threshold=flow_threshold, channels=[0, 0],
                                   diameter=z_diameter)[0]
    yz_masks = np.array(yz_masks).transpose(1, 0, 2)
    print("yz masks computed")

    print("computing xz masks")
    xz_masks = model.eval(list(img.transpose(2, 1, 0)), flow_threshold=flow_threshold, channels=[0, 0],
                                   diameter=z_diameter)[0]
    xz_masks = np.array(xz_masks).transpose(2, 1, 0)
    print("xz masks computed")

    print("Cellstitch combining masks")
    cellstitch_masks = full_stitch(xy_masks, yz_masks, xz_masks)

    # %% md
    # (4). Save the Stitching results:
    # %%
    io.imsave(os.path.join(output_path, output_filename), cellstitch_masks)
    print("Segmentation mask saved successfully")
    # %% md
print("Segmentation completed in {} seconds".format(time.time() - tic))
