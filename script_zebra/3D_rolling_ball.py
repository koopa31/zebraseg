import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm


def rolling_ball_background_subtraction(image, radius):
    """
    Perform background subtraction using the rolling ball algorithm.
    Adapted for both 2D and 3D images.
    :param image: Input grayscale image (2D or 3D).
    :param radius: Radius of the rolling ball.
    :return: Image with background subtracted.
    """
    return rolling_ball_background_subtraction_2d_gpu(image, radius)


def rolling_ball_background_subtraction_2d_gpu(image, radius):
    """
    Perform background subtraction using the rolling ball algorithm in 2D on GPU.
    :param image: Input 2D grayscale image.
    :param radius: Radius of the rolling ball.
    :return: Image with background subtracted.
    """
    image_gpu = cp.asarray(image)
    ball_gpu = create_rolling_ball_2d_gpu(radius)
    opened_gpu = ndi.grey_opening(image_gpu, footprint=ball_gpu)
    subtracted_gpu = image_gpu - opened_gpu
    subtracted = cp.asnumpy(subtracted_gpu)
    return subtracted


def create_rolling_ball_2d_gpu(radius):
    """
    Create a rolling ball structuring element in 2D on GPU.
    :param radius: Radius of the ball.
    :return: 2D structuring element (kernel).
    """
    L = 2 * radius + 1
    X, Y = cp.ogrid[:L, :L]
    dist_from_center = cp.sqrt((X - radius)**2 + (Y - radius)**2)
    ball = dist_from_center <= radius
    return ball.astype(cp.uint8)


def process_folder(input_folder, output_folder, radius):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all .tif files in the input folder
    image_files = glob.glob(os.path.join(input_folder, "*.TIF"))

    for image_file in tqdm(image_files, desc="Processing images"):
        print(f"Processing {image_file}")
        image = imread(image_file)

        # Check if the image is 3D
        if image.ndim == 3:
            processed_slices = []
            for i in tqdm(range(image.shape[0]), desc="Processing slices", leave=False):
                slice_2d = image[i, :, :]
                processed_slice = rolling_ball_background_subtraction_2d_gpu(slice_2d, radius)
                processed_slices.append(processed_slice)
            processed_image = np.stack(processed_slices, axis=0)
        else:
            processed_image = rolling_ball_background_subtraction_2d_gpu(image, radius)

        # Save the processed image to the output folder
        output_file = os.path.join(output_folder, os.path.basename(image_file))
        imsave(output_file, processed_image)
        print(f"Saved processed image to {output_file}")

    print("All images processed.")


input_folder = "/home/clement/Images/retraining_omp_48hpf"
output_folder = os.path.join(input_folder, "clean")
radius = 25

process_folder(input_folder, output_folder, radius)


