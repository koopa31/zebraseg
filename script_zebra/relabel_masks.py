import os
import numpy as np
from skimage import measure, io


def relabel_image(image):
    """Relabel an image so that the labels are consecutive integers starting from 1."""
    labeled_image, num_labels = measure.label(image, return_num=True, connectivity=2)
    return labeled_image, num_labels
def process_images(input_folder, output_folder):
    """Process all images in the input_folder, relabel them, and save them to output_folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".tif"):  # Add more extensions if needed
            image_path = os.path.join(input_folder, filename)
            image = io.imread(image_path)  # Read the image

            if image is not None:
                relabeled_image, num_labels = relabel_image(image)
                output_path = os.path.join(output_folder, filename)
                io.imsave(output_path, relabeled_image.astype(np.uint16))  # Save as 16-bit image

                # Print properties of labeled regions
                props = measure.regionprops(relabeled_image)
                for prop in props:
                    print(f"Label: {prop.label}, Area: {prop.area}, Centroid: {prop.centroid}")


# Set your input and output directories
input_folder = "/home/clement/Images/20240513_OmpYFP_84mKate_O40_05u_90u_12ss/deconv_custom_model/eon_classif/Masks"
output_folder = ("/home/clement/Images/20240513_OmpYFP_84mKate_O40_05u_90u_12ss/deconv_custom_model/eon_classif/"
                 "relabeled_masks")

process_images(input_folder, output_folder)
