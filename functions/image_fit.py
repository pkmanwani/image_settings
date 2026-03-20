import sys

sys.path.append('../')
from utils.fitting_functions import get_image, select_roi, smooth_saturated_values, filter_bright_circle_and_fit, \
    compute_beam_parameters, plot_2d_gaussian_overlay, find_threshold_crossing, calibrate_resolution
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.circle_detection import ScreenFinder


def image_fit(file_path, roi=True, sigma_size=3, get_res=False, mask_every_image=False,
              debug=False, calc_jitter=False):
    """
    Load a TIFF image and compute beam parameters with optional ROI selection and masking.

    Args:
        file_path (str): Path to the image file.
        roi (bool): Whether to allow manual ROI selection.
        sigma_size (int): Gaussian smoothing size.
        get_res (bool): Whether to calculate resolution from calibration.
        mask_every_image (bool): Force masking on every image.
        debug (bool): Save intermediate images.
        calc_jitter (bool): Compute RMS and errors over multiple images (if applicable).

    Returns:
        params (ndarray) or (rms, errors) or None
    """
    # Load image
    image, _ = get_image(file_path)
    calibration_file = "/Users/pratikmanwani/Documents/mithra_experimental_run/laser_spot_2.tiff"
    known_radius_mm = 1
    i = 0

    # Calibration
    res = calibrate_resolution(calibration_file, known_radius_mm)

    # Prepare directories
    base_dir = os.path.join(os.path.split(file_path)[0], os.path.splitext(os.path.basename(file_path))[0])
    masked_images_dir = os.path.join(base_dir, 'masked_images')
    filtered_images_dir = os.path.join(base_dir, 'filtered_images')
    threshold_images_dir = os.path.join(base_dir, 'threshold_images')
    overlayed_images_dir = os.path.join(base_dir, 'overlayed_images')

    if debug:
        os.makedirs(masked_images_dir, exist_ok=True)
        os.makedirs(filtered_images_dir, exist_ok=True)
        os.makedirs(threshold_images_dir, exist_ok=True)
    os.makedirs(overlayed_images_dir, exist_ok=True)

    results = []
    res_mm = None
    mask_switch = False

    if mask_every_image:
        mask_switch = False

    # Optional resolution via user
    while get_res:
        c1 = ScreenFinder(image)
        center = c1.circle.center
        radius = c1.circle.radius
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= (radius * 0.96) ** 2
        print(f"Circle detected at ({center[0]}, {center[1]}) with radius {radius}.")
        yag_radius = 25
        res_mm = yag_radius / c1.circle.radius
        user_res = input(f"Resolution per pixel in mm is {res_mm}. Is that fine (y/n/file)?")
        if user_res == 'y':
            res = res_mm
            get_res = False
        elif user_res == 'file':
            get_res = False
            continue

    # ROI selection
    if roi and not mask_switch:
        print("ROI mode is enabled. Please select an ROI.")
        mask = select_roi(image)
        if mask is not None:
            print("Processing selected ROI.")
            mask_switch = True
        else:
            print("No ROI selected. Using automated circle detection.")

    # Automatic circle detection if no ROI
    if not mask_switch:
        if res_mm is None:
            mask, center, radius = filter_bright_circle_and_fit(image)
            if center is not None and radius is not None:
                print(f"Circle detected at ({center[0]}, {center[1]}) with radius {radius}.")
                mask_switch = True
            else:
                user_input = input("No circle detected. Would you like to select an ROI? (y/n): ").strip().lower()
                if user_input == 'y':
                    mask = select_roi(image)
                    if mask is not None:
                        print("Processing selected ROI.")
                        mask_switch = True
                else:
                    print("No ROI or circle detected. Skipping image.")
        else:
            print("Using resolution circle")

    if mask_switch:
        print("Mask enabled. Processing image")
        masked_image = np.ma.masked_array(image, mask=~mask)
        # Convert masked array to regular array for filtering/plotting
        masked_image_filled = masked_image.filled(0)
        filtered_image = smooth_saturated_values(masked_image_filled, sigma_size)

        if debug:
            plt.imsave(os.path.join(masked_images_dir, f"masked_image_{i}.png"),
                       masked_image_filled, cmap='viridis')
            plt.imsave(os.path.join(filtered_images_dir, f"filtered_image_{i}.png"),
                       filtered_image, cmap='viridis')

        threshold_value = find_threshold_crossing(masked_image, threshold_images_dir, i, debug)
        print("Threshold value:", threshold_value)

        if threshold_value is not None:
            params = compute_beam_parameters(masked_image, threshold_value, res)
            print('Parameters (Cx,Cy,Sx,Sy,Sxy, Rx,Ry,Rxy,angle,res):', params)
            results.append(params)
            plot_2d_gaussian_overlay(overlayed_images_dir, filtered_image, i, *params)

            if not calc_jitter:
                return params, np.zeros_like(params)

    # Jitter calculation
    if calc_jitter and results:
        results = np.array(results)
        results = results[~np.isnan(results).any(axis=1)]
        if len(results) > 1:
            rms = np.sqrt(np.mean(results ** 2, axis=0))
            errors = np.std(results, axis=0) / np.sqrt(len(results))
            return rms, errors
        print("No valid results found for jitter calculation.")

    return None
if __name__ == "__main__":
    file_path = "/Users/pratikmanwani/Documents/mithra_experimental_run/beam_images/without plasma_200us exposure.tiff"
    rms, errors = image_fit(file_path,roi=True,get_res=False,mask_every_image=False,debug=True,calc_jitter=False)
    print(rms)
    print(errors)
