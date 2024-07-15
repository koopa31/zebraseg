import cv2
from skimage.color import rgb2gray
from skimage.filters import threshold_yen
import fil_finder
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.filters import threshold_yen
from skimage.io import imread
from dsepruning import skel_pruning_DSE
from scipy.ndimage import distance_transform_edt
from skimage import io, color, restoration, morphology, filters
from scipy.ndimage import convolve

import cv2
import numpy as np
from skimage.morphology import ball, binary_closing


def rolling_ball_background_subtraction(image, radius):
    """
    Perform background subtraction using the rolling ball algorithm.
    :param image: Input grayscale image.
    :param radius: Radius of the rolling ball.
    :return: Image with background subtracted.
    """
    # Create the rolling ball (structuring element)
    ball = create_rolling_ball(radius)

    # Dilate the image using the rolling ball
    dilated = cv2.dilate(image, ball)

    # Subtract the background (dilated image) from the original image
    subtracted = dilated - image

    return subtracted

def create_rolling_ball(radius):
    """
    Create a rolling ball structuring element.
    :param radius: Radius of the ball.
    :return: Structuring element (kernel).
    """
    L = 2 * radius + 1
    X, Y = np.ogrid[:L, :L]
    dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2)

    ball = dist_from_center <= radius
    return ball.astype(np.uint8)


from scipy.ndimage import gaussian_filter

def sharpen(image):
    # Noyau de convolution pour le sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 12, -1],
                       [-1, -1, -1]])

    # Appliquer le filtre de convolution
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image

image = imread("/home/clement/Images/GadalStudent-1.tif")
#image = rgb2gray(image)

# Définir le rayon du rolling ball
ball_radius = 1


# Appliquer la soustraction de fond avec un rolling ball et un élément structurant de taille 1

sharp = sharpen(image)
result = rolling_ball_background_subtraction(sharp, 2)


# Afficher les images côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Image originale')
axes[0].axis('off')
axes[1].imshow(result, cmap='gray')
axes[1].set_title('Résultat du sharpening + rolling ball')
axes[1].axis('off')
plt.show()
image = result.copy()

yen_threshold = threshold_yen(image)
binary_image = (image > yen_threshold).astype("uint8")
plt.imshow(binary_image*255, cmap='gray')
plt.title("filtrage de Yen")
plt.show()

labeled_image, num_features = ndimage.label(binary_image)

# Trouver les tailles des objets
object_slices = ndimage.find_objects(labeled_image)
sizes = [np.sum(labeled_image[slice] == i + 1) for i, slice in enumerate(object_slices)]

# Identifier l'étiquette du plus grand objet
largest_object_label = np.argmax(sizes) + 1

# Créer une image binaire avec uniquement le plus grand objet
largest_object = (labeled_image == largest_object_label).astype(np.uint8) * 255

# Afficher les résultats
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Image Originale')
plt.imshow(binary_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Image Étiquetée')
plt.imshow(labeled_image, cmap='nipy_spectral')

plt.subplot(1, 3, 3)
plt.title('Plus Gros Objet')
plt.imshow(largest_object, cmap='gray')

plt.show()

# Squelettiser l'objet
skeleton = morphology.skeletonize(largest_object)

plt.imshow(skeleton); plt.title("squelette"); plt.show()

res = skel_pruning_DSE(skeleton, distance_transform_edt(skeleton, return_indices=False, return_distances=True), 25)
plt.imshow(res);plt.show()