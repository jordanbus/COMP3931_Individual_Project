import numpy as np
import cv2

# Randomly apply a data augmentation technique to images and mask
def perform_data_augmentation(images, mask):

    # Randomly select an augmentation technique
    augmentation_type = np.random.choice(
        ['rotate', 'flip_horizontal', 'flip_vertical', 'adjust_brightness', 'zoom'])
    
    # Only apply augmentation to slices containing tumor classes
    if np.max(mask) > 0:
        augmented_images = []
        if augmentation_type == 'rotate':
            # Apply rotation
            angle = np.random.uniform(-15, 15)
            for image in images:
                augmented_images.append(rotate_image(image, angle))
            mask = rotate_image(mask, angle)

        elif augmentation_type == 'flip_horizontal':
            # Apply horizontal flip
            for image in images:
                augmented_images.append(cv2.flip(image, 1))
            mask = cv2.flip(mask, 1)
        elif augmentation_type == 'flip_vertical':
            # Apply vertical flip
            for image in images:
                augmented_images.append(cv2.flip(image, 0))
            mask = cv2.flip(mask, 0)
        elif augmentation_type == 'adjust_brightness':
            # Apply brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            for image in images:
                augmented_images.append(
                    adjust_brightness(image, brightness_factor))
        elif augmentation_type == 'zoom':
            # Random zoom
            zoom_factor = np.random.uniform(0.9, 1.1)
            for image in images:
                augmented_images.append(zoom_image(image, zoom_factor))
            mask = zoom_image(mask, zoom_factor)

        return augmented_images, mask

    # Return original images and mask for non-tumor slices
    return images, mask


# Rotate the image by given angle
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))


# Zoom image by given factor
def zoom_image(image, factor):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, factor)
    return cv2.warpAffine(image, M, (cols, rows))

# Adjust brightness of image by given factor
def adjust_brightness(image, factor):
    # Convert image to float32 for arithmetic operations
    image = image.astype(np.float32)

    # Adjust brightness of each channel separately
    for i in range(image.shape[-1]):
        image[..., i] = image[..., i] * factor

    # Clip pixel values to [0, 255]
    image = np.clip(image, 0, 255)

    # Convert image back to uint8
    image = image.astype(np.uint8)

    return image
