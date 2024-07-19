"""
This script is dedicated to the segmentation of cell membranes in Zebrafish embryos acquired in 3D with a spinning disk
microscope.

Author: Clément Cazorla
Date: 05/15/2024
Version: 1.0

It segments slice by slice in all 3 directions using Cellpose 3.0 (or any 2D cell segmentation model) and recombines the
segmentation masks using Cellstitch to get the final 3D mask.
"""
import cupy as cp
from scipy import ndimage as ndi
from skimage.io import imread, imsave
import numpy as np
from skimage.filters import threshold_otsu
from cupyx.scipy import ndimage as ndi
import pyclesperanto as cle
import cucim.skimage as cuci_ski
from skimage.measure import regionprops
import argparse
import os
from tqdm import tqdm


# Define a function to keep the largest component
def keep_largest_component(binary_image):
    # Label connected components
    labeled_array, num_features = ndi.label(binary_image)

    if num_features == 0:
        return binary_image

    # Find the largest component
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0
    largest_component_label = component_sizes.argmax()
    largest_component = (labeled_array == largest_component_label)

    return largest_component

# Load the 3D image using CuCIM
#image = imread("/home/clement/Images/retraining_omp_48hpf/rollingballed.tif")

parser = argparse.ArgumentParser(description='Parameters to set to segment zebrafish 3D olfactive epithelium.')
parser.add_argument('-p', type=str, help='Folder path containing the images to segment')

args = parser.parse_args()

path = args.p
output_folder = os.path.join(path, "detection_rosette")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

fichiers_tiff = sorted([f for f in os.listdir(path) if f.endswith('.tif') or f.endswith('.tiff')
                            or f.endswith('.TIF')])

for fichier in tqdm(fichiers_tiff, "Processing images"):
    image = imread(os.path.join(path, fichier))

    # Convert to grayscale if necessary
    gray_image = image.copy()

    # Apply mean thresholding
    thresh = threshold_otsu(gray_image, nbins=2048)
    binary_mask = gray_image > thresh

    # Remove small objects and fill holes on GPU using CuCIM
    cleaned_mask = cuci_ski.morphology.remove_small_objects(cp.array(binary_mask), min_size=10000)
    cleaned_mask = cuci_ski.morphology.remove_small_holes(cleaned_mask, area_threshold=5000)

    # Apply morphological closing to smooth object boundaries on GPU using CuCIM
    scale_factor = 0.25
    cleaned_mask = cuci_ski.transform.rescale(cleaned_mask, scale_factor)
    cleaned_mask = ndi.binary_closing(cleaned_mask, cuci_ski.morphology.ball(15))

    cleaned_mask = cle.voronoi_otsu_labeling(cp.asnumpy(cleaned_mask).astype("uint8"), spot_sigma=9, outline_sigma=1)

    properties = regionprops(cleaned_mask)

    sorted_regions = sorted(properties, key=lambda x: x.extent, reverse=True)

    selected_mask = np.zeros_like(cleaned_mask, dtype=bool)
    rosette_region_coords = sorted_regions[0].coords
    selected_mask[rosette_region_coords[:, 0], rosette_region_coords[:, 1], rosette_region_coords[:, 2]] = 1

    selected_mask = cp.asnumpy(cuci_ski.transform.rescale(cp.array(selected_mask), 1/scale_factor, order=0))

    imsave(os.path.join(output_folder, fichier), selected_mask)
