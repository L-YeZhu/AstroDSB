# --------------------------------------------------------------------------------------------------
# Core code for Astro-DSB for astrophysical observational inversion, for submission review only
# --------------------------------------------------------------------------------------------------


import numpy as np


def crop_image(img, patch_size=128, step=8):

    """Crop an image into overlapping 128x128 patches with a step size of 8."""

    h, w = img.shape[:2]
    patches = []
    positions = []

    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = img[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            positions.append((x,y))
    return patches, positions, (h, w)


def merge_patches(patches, positions, img_size, patch_size=128, step=8):

    """Merge overlapping 128x128 patches back into an image of size img_size."""
    h, w = img_size
    merged_img = np.zeros((h,w), dtype=np.float32)
    # Assuming 3 channels
    count_map = np.zeros((h,w), dtype=np.float32)

    for patch, (x, y) in zip(patches, positions):
        merged_img[y:y+patch_size, x:x+patch_size] += patch
        count_map[y:y+patch_size,x:x+patch_size] += 1


    # Avoid division by zero

    count_map[count_map == 0] = 1

    merged_img /= count_map 
    # Normalize by the number of overlapping patches


    # Crop the overshoot at the boundary

    merged_img = merged_img[:h, :w]

    return merged_img



# Example usage
patches, positions, img_size = crop_image(data_resample)

reconstructed_img = merge_patches(patches, positions, img_size)