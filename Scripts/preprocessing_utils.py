import numpy as np
from skimage.transform import resize
from skimage.color import gray2rgb

# Define target size, consistent for all datasets
TARGET_SIZE = (80, 80)

def preprocess_images(raw_images_list, target_size):
    """Resizes, standardizes channels, and normalizes images from a list."""
    preprocessed_images = []
    for img in raw_images_list:
        # Handle grayscale images by converting to RGB
        if img.ndim == 2:
            img = gray2rgb(img)
        # Handle RGBA images by removing the alpha channel
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        
        # Resize and normalize pixel values to [0, 1]
        img_resized = resize(img, target_size, anti_aliasing=True)
        preprocessed_images.append(img_resized.astype(np.float32))
    return np.array(preprocessed_images)
