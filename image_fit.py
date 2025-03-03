### main.py
from utils.fitting_functions import get_images, select_roi, smooth_saturated_values, filter_bright_circle_and_fit, \
    calculate_rms_corrected, calculate_rms_vectorized
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter, rotate
from utils.fitting_methods import fit_gaussian_linear_background
import cv2
import numpy as np
from skimage.measure import ransac, CircleModel
from scipy.signal import find_peaks
# Example usage
# filtered_image, circle_params, inliers = filter_bright_circle_and_fit(image)
import matplotlib.patches as patches

import json

def save_beam_parameters(file_path, Cx, Cy, Sx, Sy, Sxy, Rx, Ry, Rxy,angle):
    """
    Saves beam parameters in a JSON file within the same folder as the input file.

    Args:
        file_path (str): Path to the input HDF5 file.
        Cx, Cy, Sx, Sy, Sxy, Rx, Ry, Rxy, angle (float): Computed beam parameters.
    """

    # Extract folder and subfolder names
    folder_name, subfolder_name = os.path.dirname(file_path).split('/')[-2:]

    # Extract file details
    file_name = os.path.basename(file_path).replace(".h5", "")
    file_parts = file_name.split("_")

    # Define JSON file path (one per folder)
    json_file_path = os.path.join(os.path.dirname(file_path), f"{subfolder_name}.json")

    # Load existing JSON data if file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Store the parameters under subfolder -> file_name structure
    if subfolder_name not in data:
        data[subfolder_name] = {}

    data[subfolder_name][file_name] = {
        "file_parts": file_parts,
        "Cx": Cx, "Cy": Cy, "Sx": Sx, "Sy": Sy, "Sxy": Sxy,
        "Rx": Rx, "Ry": Ry, "Rxy": Rxy, "angle" : angle
    }

    # Save the updated data
    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved beam parameters in {json_file_path}")

def plot_2d_gaussian_overlay(overlay_images_dir,image, Cx, Cy, Rx, Ry, Rxy,angle,index):
    """
    Plots the image with the 2D Gaussian contour overlaid.

    Args:
        image (ndarray): The original image.
        Cx, Cy (float): Center of the Gaussian.
        Rx, Ry (float): RMS beam sizes.
        Rxy (float): Shear term.
    """
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    cov_matrix = np.array([[Rx ** 2, Rxy], [Rxy, Ry ** 2]])
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Get the major and minor axis (4 * std dev)
    major_axis = 4 * np.sqrt(eigvals[1])  # Larger eigenvalue -> major axis
    minor_axis = 4 * np.sqrt(eigvals[0])  # Smaller eigenvalue -> minor axis
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))  # Rotation angle

    # Plot the image
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap='inferno', origin='upper')

    # Draw ellipse
    ellipse = patches.Ellipse((Cx, Cy), width=major_axis, height=minor_axis,
                              angle=angle, edgecolor='cyan', facecolor='none', linewidth=2)
    plt.gca().add_patch(ellipse)

    plt.colorbar(label='Density')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    #plt.title('Image with 2D Gaussian Overlay')
    #plt.show()
    plt.savefig(f"{overlay_images_dir}/overlay_{index}.png")
    plt.close()

def compute_beam_parameters(masked_image, threshold,compute_covariance=True):
    """
    Computes beam parameters Cx, Cy, Sx, Sy, Rx, and Ry at a given threshold.

    Args:
        masked_image (ndarray): The input masked image.
        threshold (float): The background threshold value.

    Returns:
        tuple: (Cx, Cy, Sx, Sy, Rx, Ry)
    """

    # Apply threshold
    temp_im = masked_image - threshold
    temp_im[temp_im < 0] = 0  # Remove negative values

    # Compute projections
    x_projection = np.sum(temp_im, axis=0)
    y_projection = np.sum(temp_im, axis=1)

    # Fit Gaussian to projections
    _, Cx, Sx, _ = fit_gaussian_linear_background(x_projection, visualize=False)
    _, Cy, Sy, _ = fit_gaussian_linear_background(y_projection, visualize=False)
    Sxy = None
    if compute_covariance:
        # Define rotation angle
        angle_projection = -45  # degrees
        theta = np.radians(angle_projection)

        # Create rotation matrix for u and v coordinates
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                    [-np.sin(theta), np.cos(theta)]])

        # Get the shape of the image
        height, width = temp_im.shape

        # Generate meshgrid for the x and y coordinates
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Stack coordinates into (N, 2) array
        coordinates = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

        # Apply rotation to the coordinates
        transformed_coords = coordinates @ rotation_matrix.T

        # Reshape back into image dimensions
        u_coords = transformed_coords[:, 0].reshape((height, width))
        v_coords = transformed_coords[:, 1].reshape((height, width))

        # Compute projections in u and v directions
        u_projection = np.sum(temp_im * u_coords, axis=0)
        v_projection = np.sum(temp_im * v_coords, axis=1)

        # Fit Gaussian to u and v projections
        _, Cu, Su, _ = fit_gaussian_linear_background(u_projection, visualize=False)
        _, Cv, Sv, _ = fit_gaussian_linear_background(v_projection, visualize=False)

        # Compute shear term
        Sxy = (Su ** 2 - Sv ** 2) / 4
    # Compute RMS sizes
    Rx, Ry, Rxy = calculate_rms_vectorized(temp_im, Cx, Cy, compute_covariance)

    angle = 0.5 * np.arctan2(2 * Rxy, Rx - Ry)
    return Cx, Cy, Sx, Sy, Sxy, Rx, Ry, Rxy, angle

def find_threshold_crossing(masked_image, threshold_images_dir, index):
    """
    Finds the threshold value where the gradient of RMS size crosses the mean gradient after reaching a maximum.

    Args:
        masked_image (ndarray): The input masked image.
        threshold_images_dir (str): Directory path to save the threshold plot.
        index (int): Index to differentiate saved plots.

    Returns:
        float or None: The threshold value where the gradient of Rx crosses the mean after the maximum.
    """

    # Generate threshold values
    thresholds = np.linspace(0, np.mean(masked_image) * 2, 30)
    Cx, Cy, Sx, Sy, Rx, Ry = np.zeros((6, len(thresholds)))

    # Iterate through thresholds
    for j, bg in enumerate(thresholds):
        Cx[j],Cy[j],Sx[j],Sy[j],_, Rx[j], Ry[j], _, _ = compute_beam_parameters(masked_image, bg)

    print("Done processing")

    # Compute derivatives
    grad_Rx = np.gradient(Rx, thresholds)
    grad_Ry = np.gradient(Ry, thresholds)

    # Compute magnitudes
    magnitude = np.sqrt(np.abs(grad_Rx*grad_Ry))
    mean_threshold = np.mean(magnitude)

    # Find the first peak
    peaks, _ = find_peaks(magnitude)

    if len(peaks) > 0:
        max_idx = peaks[0]  # First detected peak
    else:
        max_idx = np.argmax(magnitude)  # Fallback to global maximum

    print(f"Using max_idx = {max_idx}")

    # Find first crossing after the peak
    crossing_idx = np.where((magnitude[max_idx:] < mean_threshold))[0]
    if len(crossing_idx) > 0:
        crossing_idx = crossing_idx[0] + max_idx  # Adjust for slicing offset
        threshold_value = thresholds[crossing_idx]
        print(f"Threshold where RMS gradient crosses mean after peak: {threshold_value}")
    else:
        threshold_value = None
        print("No crossing found after maximum.")

    # Plot for visualization
    plt.plot(thresholds, np.sqrt(Sx*Sy) / np.max(np.sqrt(Sx*Sy)), label="Normalized |Sxy|")
    plt.plot(thresholds, np.sqrt(Rx*Ry) / np.max(np.sqrt(Rx*Ry)), label="Normalized |Rxy|")
    plt.plot(thresholds, magnitude, label="|dRx/dThresholds|")
    plt.axhline(mean_threshold, color='red', linestyle='--', label="Mean Threshold")
    if threshold_value is not None:
        plt.axvline(threshold_value, color='green', linestyle='--', label="Threshold Crossing")
    plt.legend()

    # Save the figure
    plt.savefig(f"{threshold_images_dir}/threshold_{index}.png")
    plt.close()

    return threshold_value

def image_fit(file_path,roi=True,window_size = 5,sigma_size = 3):
    images,res = get_images(file_path)
    n_images=len(images)
    # Directories to store images
    masked_images_dir = os.path.join(file_path.split('/')[0],file_path.split('/')[1], 'masked_images', file_path.split('/')[-1].split('.')[0])
    filtered_images_dir = os.path.join(file_path.split('/')[0],file_path.split('/')[1], 'filtered_images', file_path.split('/')[-1].split('.')[0])
    threshold_images_dir = os.path.join(file_path.split('/')[0],file_path.split('/')[1], 'threshold_images', file_path.split('/')[-1].split('.')[0])
    overlayed_images_dir = os.path.join(file_path.split('/')[0],file_path.split('/')[1], 'overlayed_images', file_path.split('/')[-1].split('.')[0])

    # Create directories if they don't exist
    os.makedirs(masked_images_dir, exist_ok=True)
    os.makedirs(filtered_images_dir, exist_ok=True)
    os.makedirs(threshold_images_dir, exist_ok=True)
    os.makedirs(overlayed_images_dir, exist_ok=True)
    # Store results for each image
    results = []
    Cx_final, Cy_final,Sx_final,Sy_final,Sxy_final,Rx_final,Ry_final,Rxy_final, angle = np.zeros((9,1))
    for i, image in enumerate(images):
        mask_switch = False
        if roi:  # If ROI mode is enabled, prompt the user to select ROI first
            print("ROI mode is enabled. Please select an ROI.")
            mask = select_roi(image)  # Replace with actual ROI selection logic

            if mask is not None:
                print("Processing selected ROI.")
                mask_switch = True
            else:
                print("No ROI selected. Proceeding to automatic circle detection.")

        if mask_switch==False:
        # If ROI is False or ROI selection failed, try automatic circle detection
            mask, center, radius = filter_bright_circle_and_fit(image)

            if center is not None and radius is not None:
                xc, yc = center
                print(f"Circle detected at ({xc}, {yc}) with radius {radius}.")
                mask_switch = True
            else:
                # If no circle is found, ask the user if they want to manually select an ROI
                user_input = input("No circle detected. Would you like to select an ROI? (y/n): ").strip().lower()
                if user_input == 'y':
                    print("User selecting ROI...")
                    mask = select_roi(image)  # Allow manual selection
                    if mask is not None:
                        print("Processing selected ROI.")
                        mask_switch=True  # Retry processing with selected ROI
                else:
                    print("No ROI or circle detected. Going to next image.")

        if mask_switch:
            print("Mask enabled. Processing image")
            masked_image = np.ma.masked_array(image, mask=~mask)
            #filter image
            filtered_image = smooth_saturated_values(masked_image, sigma_size)

            # Save the masked image
            masked_image_path = os.path.join(masked_images_dir, f"masked_image_{i}.png")
            plt.imsave(masked_image_path, masked_image,cmap='viridis')  # Save masked image as PNG

            # Save the filtered image
            filtered_image_path = os.path.join(filtered_images_dir, f"filtered_image_{i}.png")
            plt.imsave(filtered_image_path, filtered_image,cmap='viridis')  # Save filtered image as PNG

            threshold_value = find_threshold_crossing(masked_image, threshold_images_dir, index=i)
            print(threshold_value)
            if threshold_value is not None:
                Cx_final, Cy_final, Sx_final, Sy_final, Sxy_final, Rx_final, Ry_final, Rxy_final, angle = compute_beam_parameters(masked_image, threshold_value)
                print('Beam parameters (Cx,Cy,Sx,Sy,Sxy, Rx,Ry,Rxy):', Cx_final, Cy_final, Sx_final, Sy_final, Sxy_final, Rx_final, Ry_final, Rxy_final, angle)
                plot_2d_gaussian_overlay(overlayed_images_dir,filtered_image,Cx_final, Cy_final, Rx_final, Ry_final, Rxy_final,angle,index=i)
                return Cx_final, Cy_final, Sx_final, Sy_final, Sxy_final, Rx_final, Ry_final, Rxy_final, angle, res
if __name__ == "__main__":
    file_path = "2025_02_19-selected/3ScreenMeasurement_LongPulse/Yag6_1739998915.h5"
    image_fit(file_path,roi=True)
