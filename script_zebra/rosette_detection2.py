"""
This script is dedicated to the segmentation of cell membranes in Zebrafish embryos acquired in 3D with a spinning disk
microscope.

Author: Clément Cazorla
Date: 05/15/2024
Version: 1.0

It segments slice by slice in all 3 directions using Cellpose 3.0 (or any 2D cell segmentation model) and recombines the
segmentation masks using Cellstitch to get the final 3D mask.
"""


"""import os
import time
import argparse

import numpy as np
import torch
import tifffile
from cellpose.models import Cellpose
from skimage.io import imread, imsave
from skimage.filters import threshold_mean
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from skimage.morphology import erosion, disk, dilation

from cellstitch.pipeline import full_stitch
# %%
# Plotting specifications
from matplotlib import rcParams
#from IPython.display import display


def keep_largest_component(binary_image):
    # Étiquetage des composants connectés
    labeled_array, num_features = label(binary_image)

    # Si aucun composant n'est trouvé, retourner l'image originale
    if num_features == 0:
        return binary_image

    # Trouver les aires de chaque composant
    component_sizes = np.bincount(labeled_array.ravel())

    # Ignorer le fond (label 0)
    component_sizes[0] = 0

    # Trouver le label du plus grand composant
    largest_component_label = component_sizes.argmax()

    # Créer une nouvelle image binaire avec seulement le plus grand composant
    largest_component = (labeled_array == largest_component_label)

    return largest_component


image = imread("/home/clement/Images/retraining_omp_48hpf/"
               "clean/20240522_OmpRFP_SoxteenGFP_H2BCER_48hpf_O40_02u_80um_boite2_suite_w1CSU-561 Em 593LP_s4.TIF")[220, :]

imsave("/home/clement/Images/slice.png", image)

th = threshold_mean(image)

mask = image > th

mask = dilation(mask, disk(3))
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.show()

mask = erosion(mask, disk(8))
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.show()
mask = dilation(mask, disk(4))

plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.show()

closed_mask = erosion(mask, disk(30))

mask = keep_largest_component(closed_mask)

mask = dilation(mask, disk(30))

plt.imshow(mask, cmap='gray')
plt.show()"""

"""import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_mean
from skimage.morphology import closing, square, remove_small_objects, binary_closing, disk
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# Load the image
image = imread("/home/clement/Images/retraining_omp_48hpf/"
               "clean/20240522_OmpRFP_SoxteenGFP_H2BCER_48hpf_O40_02u_80um_boite2_suite_w1CSU-561 Em 593LP_s4.TIF")[220, :]

# Convert to grayscale if it's not already
gray_image = image.copy()


# Apply Otsu's thresholding
thresh = threshold_mean(gray_image)
binary_mask = gray_image > thresh

# Remove small objects and fill holes
cleaned_mask = remove_small_objects(binary_mask, min_size=500)
cleaned_mask = ndi.binary_fill_holes(cleaned_mask)

# Apply a morphological closing to smooth the object boundaries
cleaned_mask = binary_closing(cleaned_mask, disk(15))

# Distance transform
distance = ndi.distance_transform_edt(cleaned_mask)

# Find local maxima
local_maxi = peak_local_max(distance, footprint=np.ones((19, 19)), labels=cleaned_mask)

# Perform watershed
markers = np.zeros_like(distance, dtype=bool)
markers[tuple(local_maxi.T)] = True
markers = ndi.label(markers)[0]
labels = watershed(-distance, markers, mask=cleaned_mask)

plt.imshow(labels)
plt.title("Labels")
plt.show()

# Select the region of interest (blob structure)
blob_label = np.argmax(np.bincount(labels.flat)[1:]) + 1  # Skipping the background label (0)
blob_mask = labels == blob_label

# Display the result
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(blob_mask, cmap='gray')
ax[1].set_title('Binary Mask of Blob Structure')
ax[2].imshow((blob_mask * image)*10, cmap='gray')
ax[2].set_title('Masked Image')
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import threshold_mean, threshold_yen
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pyclesperanto_prototype as cle
from scipy import ndimage as ndi
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_mean
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pyclesperanto_prototype as cle
import cucim.skimage as cuci_ski
from cucim.skimage.filters import threshold_otsu

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

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_mean, threshold_yen, threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
from skimage.feature import peak_local_max
from cupyx.scipy import ndimage as ndi
import pyclesperanto as cle
import cucim.skimage as cuci_ski
from skimage.measure import label, regionprops

# Load the 3D image using CuCIM
image = imread("/home/clement/Images/retraining_omp_48hpf/rollingballed.tif")

# Downsize the image to speed up processing
#scaling_factor = 0.5  # Adjust this factor as needed
#downsampled_image = ndi.zoom(image, zoom=scaling_factor, order=1)

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

#plt.imshow(selected_mask)[50, :]; plt.show()
imsave("/home/clement/Images/test.tif", selected_mask)

"""import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import scipy.ndimage as ndi
from cellpose.models import Cellpose


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

# Load the 3D image
image = imread("/home/clement/Images/retraining_omp_48hpf/"
               "clean/20240522_OmpRFP_SoxteenGFP_H2BCER_48hpf_O40_02u_80um_boite2_suite_w1CSU-561 Em 593LP_s4.TIF")

zoom_factor = 0.25
#image = ndi.zoom(image, zoom=zoom_factor, order=1)

model = Cellpose(gpu=True, model_type="cyto3")

mask = model.eval(image, batch_size=32, do_3D=True, diameter=250)[0]

#mask = ndi.zoom(mask, zoom=1/zoom_factor, order=1)
mask = keep_largest_component(mask)
plt.imshow(mask[200, :])
plt.title("Mask")
plt.show()

imsave("/home/clement/Images/seg_cp_3D_rosette.tif", mask)"""