import os
import tifffile as tiff
import numpy as np

# Dossier contenant les fichiers TIFF
input_dir = '/home/clement/Images/Essai trackmate/'
output_dir = '/home/clement/Images/Essai trackmate/separated_channels/'

# Créez le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Liste tous les fichiers .tif dans le dossier d'entrée
tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]


# Fonction pour extraire et enregistrer les canaux
def save_channels(file_path, output_dir):
    # Lire le fichier TIFF
    img = tiff.imread(file_path)

    # Vérifiez si l'image a deux canaux
    if img.shape[0] != 2:
        print(f"File {file_path} does not contain 2 channels, skipping.")
        return

    # Sépare les canaux
    channel_1 = img[0, :]
    channel_2 = img[1, :]

    # Nom du fichier sans extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Enregistre le canal 1
    tiff.imsave(os.path.join(output_dir, f"{file_name}_channel_1.tif"), channel_1)
    # Enregistre le canal 2
    tiff.imsave(os.path.join(output_dir, f"{file_name}_channel_2.tif"), channel_2)


# Traiter chaque fichier TIFF
for tiff_file in tiff_files:
    file_path = os.path.join(input_dir, tiff_file)
    save_channels(file_path, output_dir)

print("Traitement terminé.")
