"""
Augmentation module for neutron star diffusion.
Migrated and cleaned up.
"""

from keras.layers import (RandomCrop, RandomFlip, RandomTranslation, 
                            RandomRotation, RandomZoom, RandomContrast,
                            RandomBrightness)
import os
import numpy as np
from skimage import io
import tensorflow as tf
import random

random_crop = RandomCrop(height=200, width=200)
random_flip = RandomFlip("horizontal_and_vertical")
random_translation = RandomTranslation(height_factor=0.2, width_factor=0.2)
random_rotation = RandomRotation(factor=0.2)
random_zoom = RandomZoom(height_factor=(-0.3, 0.2), width_factor=(-0.3, 0.2))
random_contrast = RandomContrast(factor=0.5)
random_brightness = RandomBrightness(factor=0.3)

# Liste mit den Funktionen
augmentations = [
    random_crop, 
    random_flip, 
    random_translation, 
    random_rotation, 
    random_zoom, 
    random_contrast, 
    random_brightness
]

# Wähle zufällig eine Augmentierung aus und wende sie auf das Bild an
def apply_random_augmentation(image):
    augmentation_function = random.choice(augmentations)
    augmented_image = augmentation_function(tf.expand_dims(image, 0))  # Batch-Dimension hinzufügen
    return tf.squeeze(augmented_image)  # Batch-Dimension entfernen


# Define the dimensions of the image
height, width = 256, 256

# Create an image with a simple gradient from black to white
def create_test_image(height, width):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    gradient = np.outer(y, x)  # Combine gradients into a 2D image
    image = np.stack([gradient, gradient, gradient], axis=-1)  # Convert to RGB
    return image

# Generate the test image
image = create_test_image(height, width)

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss, 0, 1)  # Ensure pixel values are in [0, 1]
    return noisy_image

# Example usage
noisy_image = add_gaussian_noise(image)



def augment_and_save(image_path, output_folder, num_augments=10):
    image = io.imread(image_path)
    datagen = apply_random_augmentation(image)
    
    image = image.reshape((1, ) + image.shape)  # Reshape for Keras
    
    for i, batch in enumerate(datagen.flow(image, batch_size=1)):
        noisy_image = add_gaussian_noise(batch[0].astype('float32'))
        save_path = os.path.join(output_folder, f"augmented_{i}.png")
        io.imsave(save_path, noisy_image)
        if i >= num_augments - 1:
            break

# Example usage
augment_and_save('path_to_your_image.png', 'output_folder')
