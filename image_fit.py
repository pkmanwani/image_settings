### main.py
from utils.fitting_functions import get_images,create_circular_mask,select_roi,smooth_saturated_values
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter, rotate

def image_fit(file_path,roi=True,window_size = 10,sigma_size = 3):
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

        #filter image
        filtered_image = denoise_median_arrayed(masked_image, window_size)
        filtered_image = smooth_saturated_values(filtered_image, sigma_size)

        # Save the masked image
        masked_image_path = os.path.join(masked_images_dir, f"masked_image_{i}.png")
        plt.imsave(masked_image_path, masked_image.filled(0), cmap='gray')  # Save masked image as PNG

        # Save the filtered image
        filtered_image_path = os.path.join(filtered_images_dir, f"filtered_image_{i}.png")
        plt.imsave(filtered_image_path, filtered_image, cmap='gray')  # Save filtered image as PNG

        thresholds = np.linspace(0, np.mean(test_img) * 2, 20)
        print(thresholds)
        Cx, Cy, Sx, Sy, Rx, Ry = np.zeros((6, len(thresholds)))
        Rx, Ry = calculate_rms_vectorized(temp_im, Cx[i], Cy[i])
        i = 0
        for bg in thresholds:
            temp_im = test_img - bg
            temp_im[temp_im < 0] = 0
            x_projection = np.sum(temp_im, axis=0)
            y_projection = np.sum(temp_im, axis=1)
            [_, Cx[i], Sx[i], _] = fit_gaussian_linear_background(x_projection, visualize=False)
            [_, Cy[i], Sy[i], _] = fit_gaussian_linear_background(y_projection, visualize=False)

            # plt.axvline(Cx[i]+Rx[i],linestyle='--')
            # plt.axhline(Cy[i]+Ry[i],linestyle='--')
            # plt.show()  # Display the plot without blocking
            # fig = plt.gcf()  # Grabs the current figure
            # plt.imshow(temp_im)
            # plt.scatter(Cx[i],Cy[i],color='red')
            # plt.axvline(Cx[i]+Sx[i],color='red')
            # plt.axvline(Cx[i]-Sx[i],color='red')
            # plt.axhline(Cy[i]-Sy[i],color='red')
            # plt.axhline(Cy[i]+Sy[i],color='red')
            # plt.axvline(Rx[i]+Cx[i])
            # plt.axvline(Cx[i]-Rx[i])
            # plt.axhline(Cy[i]-Ry[i])
            # plt.axhline(Cy[i]+Ry[i])
            # display(fig)
            # plt.clf()
            i = i + 1
        print('done')
        plt.plot(thresholds, Sx, label='Sx')
        plt.plot(thresholds, Sy, label='Sy')
        plt.plot(thresholds, Rx, label='RMSX')
        plt.plot(thresholds, Ry, label='RMSY')
        # .plot(thresholds,np.gradient(Rx,thresholds), label='RMSX_prime')
        plt.legend()


        """
        Sx, Sy, correlation = compute_statistics(filtered_image)
        plot_projections(filtered_image)
    
        save_results("results.json", Sx, Sy, correlation)"""


if __name__ == "__main__":
    file_path = "2025_02_19-selected/3ScreenMeasurement_LongPulse/SlitYag_1739998814.h5"
    image_fit(file_path)
