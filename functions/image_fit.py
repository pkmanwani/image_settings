### main.py
import sys
sys.path.append('../')
from utils.fitting_functions import get_images, select_roi, smooth_saturated_values, filter_bright_circle_and_fit,\
compute_beam_parameters, plot_2d_gaussian_overlay, find_threshold_crossing
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.circle_detection import ScreenFinder
import json


def image_fit(file_path,roi=True,sigma_size = 3,get_res=False,mask_every_image=False,debug=False,calc_jitter=False):
    images,res = get_images(file_path)
    n_images=len(images)
    # Directories to store images
    masked_images_dir = os.path.join(os.path.split(file_path)[0], file_path.split('/')[-1].split('.')[0],'masked_images')
    filtered_images_dir = os.path.join(os.path.split(file_path)[0], file_path.split('/')[-1].split('.')[0],'filtered_images')
    threshold_images_dir = os.path.join(os.path.split(file_path)[0], file_path.split('/')[-1].split('.')[0],'threshold_images')
    overlayed_images_dir = os.path.join(os.path.split(file_path)[0], file_path.split('/')[-1].split('.')[0],'overlayed_images')
    # Create directories if they don't exist
    if debug==True:
        os.makedirs(masked_images_dir, exist_ok=True)
        os.makedirs(filtered_images_dir, exist_ok=True)
        os.makedirs(threshold_images_dir, exist_ok=True)
    os.makedirs(overlayed_images_dir, exist_ok=True)
    # Store results for each image
    results = []  # Store results for each image
    res_mm = None
    mask_switch = False
    for i, image in enumerate(images):
        if mask_every_image==True:
            mask_switch=False
        while get_res == True:
            c1 = ScreenFinder(image)
            center = c1.circle.center
            radius = c1.circle.radius
            h_full, w_full = image.shape
            # Create a grid of coordinates
            y, x = np.ogrid[:h_full, :w_full]
            mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= (radius*0.96)** 2
            print(f"Circle detected at ({center[0]}, {center[1]}) with radius {radius}.")
            yag_radius = 25 #mm
            res_mm = yag_radius / c1.circle.radius
            print(f"YAG radius assumed to be {res_mm} mm.")
            user_res = input(f"Resolution per pixel in mm is {res_mm}. Is that fine (y/n/file)?")
            if user_res == 'y':
                res=res_mm
                get_res = False
            elif user_res == 'file':
                get_res = False
                continue
        if roi and not mask_switch:  # If ROI mode is enabled, prompt the user to select ROI first
            print("ROI mode is enabled. Please select an ROI.")
            mask = select_roi(image)  # Replace with actual ROI selection logic
            if mask is not None:
                print("Processing selected ROI.")
                mask_switch = True
            else:
                print("No ROI selected. Using automated circle detection.")
        if mask_switch==False:
            if res_mm is None:
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
            else:
                print("Using resolution circle")

        if mask_switch:
            print("Mask enabled. Processing image")
            masked_image = np.ma.masked_array(image, mask=~mask)
            #filter image
            filtered_image = smooth_saturated_values(masked_image, sigma_size)

            if debug == True:
                # Save the masked image
                masked_image_path = os.path.join(masked_images_dir, f"masked_image_{i}.png")
                plt.imsave(masked_image_path, masked_image,cmap='viridis')  # Save masked image as PNG

                # Save the filtered image
                filtered_image_path = os.path.join(filtered_images_dir, f"filtered_image_{i}.png")
                plt.imsave(filtered_image_path, filtered_image,cmap='viridis')  # Save filtered image as PNG

            threshold_value = find_threshold_crossing(masked_image, threshold_images_dir, i,debug)
            print(threshold_value)
            if threshold_value is not None:
                params = compute_beam_parameters(masked_image, threshold_value,res)
                print('Parameters (Cx,Cy,Sx,Sy,Sxy, Rx,Ry,Rxy,angle,res):', params)
                results.append(params)
                plot_2d_gaussian_overlay(overlayed_images_dir, filtered_image, i,*params)

                if not calc_jitter:
                    return params, np.zeros_like(params)  # Return first valid image's result, error set to None

    if calc_jitter:
        if results:
            results = np.array(results)
            results = results[~np.isnan(results).any(axis=1)]  # Remove NaN rows
            if len(results) > 1:
                rms = np.sqrt(np.mean(results ** 2, axis=0))
                errors = np.std(results, axis=0) / np.sqrt(len(results))
                return rms, errors
        print("No valid results found for jitter calculation.")

    return None  # Explicitly return None when no valid results exist

if __name__ == "__main__":
    file_path = "../beamlines/awa/2025_02_19-selected/ThreeScreenAfterFlat/Yag6_1740007655.h5"
    rms, errors = image_fit(file_path,roi=True,get_res=False,mask_every_image=False,debug=True,calc_jitter=True)
    print(rms)
    print(errors)
