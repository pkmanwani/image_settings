### main.py
from utils.fitting_functions import get_images,create_circular_mask,select_roi,denoise_median_arrayed
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter, rotate

def image_fit(file_path,roi=True,denoise_size=3):
    images,res = get_images(file_path)
    # Directories to store images
    masked_images_dir = 'masked_images'
    filtered_images_dir = 'filtered_images'

    # Create directories if they don't exist
    os.makedirs(masked_images_dir, exist_ok=True)
    os.makedirs(filtered_images_dir, exist_ok=True)
    # Store results for each image
    results = []

    for i, image in enumerate(images):
        # Create a circular mask
        mask, center, radius = create_circular_mask(image)
        if roi:
            masked_image = select_roi(image)
            if masked_image is None:
                print("No ROI selected.")
                if i==0:
                    print("Choosing circular mask with center {center} and radius {radius}.".format(
                    center=center, radius=radius))
                masked_image = np.ma.masked_array(image, mask=~mask)
        else:
            if i==0:
                print("ROI turned off. Choosing circular mask with center {center} and radius {radius}.".format(center=center, radius=radius))
            masked_image = np.ma.masked_array(image, mask=~mask)

        # Save the masked image
        masked_image_path = os.path.join(masked_images_dir, f"masked_image_{i}.png")
        plt.imsave(masked_image_path, masked_image.filled(0), cmap='gray')  # Save masked image as PNG

        # Apply denoising
        filtered_image = denoise_median_arrayed(masked_image.filled(0), size=denoise_size)

        # Save the filtered image
        filtered_image_path = os.path.join(filtered_images_dir, f"filtered_image_{i}.png")
        plt.imsave(filtered_image_path, filtered_image, cmap='gray')  # Save filtered image as PNG

        """
        Sx, Sy, correlation = compute_statistics(filtered_image)
        plot_projections(filtered_image)
    
        save_results("results.json", Sx, Sy, correlation)"""


if __name__ == "__main__":
    file_path = "2025_02_19-selected/3ScreenMeasurement_LongPulse/SlitYag_1739998814.h5"
    image_fit(file_path)
