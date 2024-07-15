import os
import numpy as np
from skimage.io import imread, imsave
import argparse

# Definition of input parameters

parser = argparse.ArgumentParser(description='Parameters to set to segment zebrafish 3D embryo membranes.')
parser.add_argument('-f', type=str, help='Folder path containing the images to segment')
parser.add_argument('-c', type=str, help='Channel to segment (wavelength)', default=561)

args = parser.parse_args()

folder = args.f
channel = args.c

output_folder = os.path.join(folder, "slices_to_retrain")

if os.path.exists(output_folder) is False:
    os.mkdir(output_folder)

# Parcourir les fichiers dans le dossier
for fichier in os.listdir(folder):
    if (fichier.endswith('.TIF') or fichier.endswith('.tif')) and channel + ' Em' in fichier:
        # Charger l'image 3D
        image_3d = imread(os.path.join(folder, fichier))

        # Calculer le signal moyen dans chaque tranche dans chaque direction
        xy_slice_mean = np.mean(np.sum(image_3d, axis=(1, 2)))  # Moyenne des tranches XY
        xz_slice_mean = np.mean(np.sum(image_3d, axis=(0, 2)))  # Moyenne des tranches XZ
        yz_slice_mean = np.mean(np.sum(image_3d, axis=(0, 1)))  # Moyenne des tranches YZ

        # Trouver l'indice de la tranche avec le signal moyen le plus fort dans chaque direction
        xy_slice_index = np.argmax(np.sum(image_3d, axis=(1, 2)))  # Slice XY
        xz_slice_index = np.argmax(np.sum(image_3d, axis=(0, 2)))  # Slice XZ
        yz_slice_index = np.argmax(np.sum(image_3d, axis=(0, 1)))  # Slice YZ

        # Extraire les tranches avec le signal moyen le plus fort
        xy_slice = image_3d[xy_slice_index, :, :]  # Slice XY
        xz_slice = image_3d[:, xz_slice_index, :]  # Slice XZ
        yz_slice = image_3d[:, :, yz_slice_index]  # Slice YZ

        # Sauvegarder les tranches avec le plus de signal
        xy_slice_name = f"XY_slice_{fichier}"
        xz_slice_name = f"XZ_slice_{fichier}"
        yz_slice_name = f"YZ_slice_{fichier}"

        imsave(os.path.join(output_folder, xy_slice_name), xy_slice)
        imsave(os.path.join(output_folder, xz_slice_name), xz_slice)
        imsave(os.path.join(output_folder, yz_slice_name), yz_slice)