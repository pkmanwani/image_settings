### functions.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from matplotlib.widgets import RectangleSelector
from .fitting_methods import fit_gaussian_linear_background
from scipy.signal import find_peaks
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
# Function to open H5 file and read images
def get_images(file_path):
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        res = f['images'].attrs['resolution']
        print(f"Resolution from file is {res}")
        if res is None:
            res = 1
            print("Setting res to 1")
    return images,res

def create_circular_mask(image,center=None, radius=None):
    h,w = image.shape
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], (w - center[0])//1.2, (h - center[1])//1.2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask, center, radius



def ransac_circle_fit(points):
    """
    Fits a circle to the given points using RANSAC.

    Parameters:
        points (np.ndarray): Array of points (x, y) to fit the circle.

    Returns:
        tuple: (x_center, y_center, radius) of the fitted circle.
    """
    # RANSAC parameters
    n_samples = 100  # Number of samples for RANSAC
    max_trials = 100  # Number of trials for RANSAC

    best_inliers = []
    best_circle = None

    for _ in range(max_trials):
        # Randomly sample points
        indices = np.random.choice(len(points), size=n_samples, replace=False)
        sample_points = points[indices]

        # Fit the circle using least squares method
        A = np.c_[sample_points[:, 0], sample_points[:, 1], np.ones(sample_points.shape[0])]
        b = sample_points[:, 0]**2 + sample_points[:, 1]**2
        params = np.linalg.lstsq(A, b, rcond=None)[0]

        # Circle center and radius
        x_center = params[0] / 2
        y_center = params[1] / 2
        radius = np.sqrt(params[2] + (x_center**2 + y_center**2))

        # Calculate distances from the fitted circle
        distances = np.sqrt((points[:, 0] - x_center)**2 + (points[:, 1] - y_center)**2)
        inliers = distances < (radius + 1)  # Tolerance for inliers

        # Update best circle if more inliers found
        if np.sum(inliers) > len(best_inliers):
            best_inliers = points[inliers]
            best_circle = (x_center, y_center, radius)
    # Return None if no valid circle was found
    if best_circle is None:
        return None

    return best_circle

def filter_bright_circle_and_fit(image):
    """
    Filters the image to retain only the bright pixels within a circular region
    centered in the image with a radius greater than h/4 and fits a circle using RANSAC.

    Parameters:
        image (np.ndarray): Input image array.

    Returns:
        tuple: (mask, (x_center, y_center, radius))
    """
    # Load image dimensions
    h_full, w_full = image.shape
    #print("Original shape:", image.shape)

    # Calculate the center and radius
    center_x, center_y = w_full // 2, h_full // 2
    radius_big = h_full // 2  # Assuming radius is greater than h/4
    radius_small = h_full // 4

    # Create a grid of coordinates
    y, x = np.ogrid[:h_full, :w_full]
    circle_mask = (x - center_x)**2 + (y - center_y)**2 >= radius_small**2
    circle_big_mask = (x - center_x)**2 + (y - center_y)**2 <= radius_big**2

    # Calculate the mean pixel value to identify bright pixels
    mean_value = np.mean(image)
    bright_mask = image > mean_value

    # Combine masks: retain only bright pixels within the circle
    final_mask = circle_mask & circle_big_mask & bright_mask

    # Extract the coordinates of bright pixels
    y_coords, x_coords = np.where(final_mask)
    points = np.column_stack((x_coords, y_coords))

    # Fit a circle to the bright points using RANSAC
    circle_params = ransac_circle_fit(points)
    if circle_params is None:
        return None, None, None
    xc, yc = circle_params[0],circle_params[1]
    radius = circle_params[2]*0.96
    circle_mask = (x - xc) ** 2 + (y - yc) ** 2 <= radius ** 2
    return circle_mask, (xc,yc), radius

# Function to let user select a draggable ROI
def select_roi(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='viridis')
    ax.set_title("Select ROI: Drag the box to select the region")

    # Store the selected ROI coordinates
    roi_coords = []

    # Define the callback function for RectangleSelector
    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        roi_coords.clear()  # Clear previous coordinates
        roi_coords.extend([x1, y1, x2, y2])

        # Optionally, print the coordinates
        print(f"ROI selected: Top-left ({x1}, {y1}), Bottom-right ({x2}, {y2})")

    # Create the RectangleSelector widget
    rectangle_selector = RectangleSelector(ax, onselect, interactive='box', useblit=True, button=[1])

    # Display the plot and wait for the user to select the ROI
    plt.show()

    # After selection, return the cropped region
    if len(roi_coords) == 4:
        x1, y1, x2, y2 = map(int, roi_coords)
        mask = np.zeros(image.shape, dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask
    else:
        return None  # In case selection wasn't made

def smooth_saturated_values(image, threshold=None, filter_size=3):
    """
    Smooths out saturated values by applying median filtering.

    Args:
        image (ndarray): 2D intensity array (image).
        threshold (float, optional): Saturation threshold. Defaults to max value in the image.
        filter_size (int): Size of the median filter (must be odd).

    Returns:
        ndarray: Image with smoothed saturated regions.
    """
    if threshold is None:
        threshold = np.max(image)  # Default to the max value in the image

    # Identify saturated pixels
    mask = image >= threshold

    # Apply median filtering to the entire image
    smoothed_image = median_filter(image, size=filter_size)

    # Replace only the saturated pixels with the smoothed values
    image[mask] = smoothed_image[mask]

    return image

def calculate_rms_vectorized(image, Cx, Cy,compute_covariance=True):
    """
    Calculate the RMS size using a vectorized approach.

    Args:
        image (ndarray): The input 2D image.
        Cx (ndarray): The x-coordinate centers for each image.
        Cy (ndarray): The y-coordinate centers for each image.

    Returns:
        Rx (ndarray): The RMS size in the x-direction for each image.
        Ry (ndarray): The RMS size in the y-direction for each image.
    """
    # Calculate distances from the center
    rows, cols = np.indices(image.shape)
    R_x = np.sqrt(np.sum((cols - Cx) ** 2 * image) / np.sum(image))
    R_y = np.sqrt(np.sum((rows - Cy) ** 2 * image) / np.sum(image))
    R_xy = None
    if compute_covariance:
        # Calculate Rxy based on covariance
        R_xy = (np.sum((cols - Cx) * (rows - Cy) * image) / np.sum(image))

    return R_x, R_y, R_xy

def calculate_rms_corrected(image):
    """
    Calculate the centroid (Cx, Cy) and RMS sizes (Rx, Ry) of an image.

    Args:
        image (ndarray): The input 2D image (intensity array).

    Returns:
        Cx (float): The x-coordinate of the centroid.
        Cy (float): The y-coordinate of the centroid.
        Rx (float): The RMS size in the x-direction.
        Ry (float): The RMS size in the y-direction.
    """

    # Generate coordinate arrays
    x_indices = np.arange(image.shape[1])  # X-coordinates
    y_indices = np.arange(image.shape[0])  # Y-coordinates

    # Compute centroid using intensity-weighted mean
    total_intensity = np.sum(image)
    Cx = np.sum(x_indices * np.sum(image, axis=0)) / total_intensity
    Cy = np.sum(y_indices * np.sum(image, axis=1)) / total_intensity

    # Compute RMS size (standard deviation weighted by intensity)
    dx = x_indices - Cx
    dy = y_indices - Cy

    # Create 2D grids of differences
    dx_grid, dy_grid = np.meshgrid(dx, dy)
    # Compute variance (weighted by intensity)
    Xnom = np.sum(image * dx_grid ** 2)
    Ynom = np.sum(image * dy_grid ** 2)

    # RMS size is the square root of variance
    Rx = np.sqrt(Xnom / total_intensity)
    Ry = np.sqrt(Ynom / total_intensity)
    return Cx,Cy,Rx,Ry


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_2d_gaussian_overlay(overlay_images_dir, image, index, *params):
    """
    Plots the image with the 2D Gaussian contour overlaid and includes X & Y projections at the edges.

    Args:
        overlay_images_dir (str): Directory to save the output image.
        image (ndarray): The original image.
        index (int): Index for saving the image.
        params (tuple): Parameters (Cx, Cy, Sx, Sy, Sxy, Rx, Ry, Rxy, angle, res).
    """
    # Unpack parameters
    Cx, Cy, Sx, Sy, Sxy, Rx, Ry, Rxy, _, res = params

    # Image dimensions in pixels
    img_height, img_width = image.shape

    # Compute bounds for cropping (Cx, Cy, Rx, Ry are in pixels)
    x_min = int(Cx - 4*Rx)
    x_max = int(Cx + 4*Rx)
    y_min = int(Cy - 4*Ry)
    y_max = int(Cy + 4*Ry)

    # Crop the image to the region of interest
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Convert parameters to mm
    Cx, Cy = Cx * res, Cy * res
    Sx, Sy, Rx, Ry = Sx * res, Sy * res, Rx * res, Ry * res
    Sxy, Rxy = Sxy * (res ** 2), Rxy * (res ** 2)

    # Image dimensions in mm
    cropped_img_height, cropped_img_width = cropped_image.shape
    cropped_img_height_mm, cropped_img_width_mm = cropped_img_height * res, cropped_img_width * res

    # Compute covariance matrices and ellipses
    cov_matrix_R = np.array([[Rx**2, Rxy], [Rxy, Ry**2]])
    eigvals_R, eigvecs_R = np.linalg.eigh(cov_matrix_R)
    major_axis_R = 4 * np.sqrt(eigvals_R[1])
    minor_axis_R = 4 * np.sqrt(eigvals_R[0])
    angle_R = np.degrees(np.arctan2(eigvecs_R[1, 1], eigvecs_R[0, 1]))

    cov_matrix_sigma = np.array([[Sx ** 2, Sxy], [Sxy, Sy ** 2]])
    eigvals_s, eigvecs_s = np.linalg.eigh(cov_matrix_sigma)
    major_axis_s = 4 * np.sqrt(eigvals_s[1])
    minor_axis_s = 4 * np.sqrt(eigvals_s[0])
    angle_s = np.degrees(np.arctan2(eigvecs_s[1, 1], eigvecs_s[0, 1]))

    # Compute projections and normalize them
    x_projection = np.sum(cropped_image, axis=0)
    y_projection = np.sum(cropped_image, axis=1)

    # Normalize projections to fit inside the cropped image space
    x_projection = (x_projection - x_projection.min()) / (x_projection.max() - x_projection.min()) * cropped_img_height * 0.2
    y_projection = (y_projection - y_projection.min()) / (y_projection.max() - y_projection.min()) * cropped_img_width * 0.2

    # Set up figure with black background
    fig, ax = plt.subplots(edgecolor='white')
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.set_facecolor('black')  # Set the axes background to black

    # Overlay X projection at the bottom edge of the image
    x_range = np.linspace(0, cropped_img_width_mm, cropped_img_width) - (Cx-x_min*res)
    ax.plot(x_range, cropped_img_height_mm - x_projection * res - (Cy-y_min*res), color='white',
            linewidth=1)  # Flipped to align with image

    # Overlay Y projection at the right edge of the image
    y_range = np.linspace(0, cropped_img_height_mm, cropped_img_height) - (Cy-y_min*res)
    ax.plot(0 + y_projection * res - (Cx-x_min*res), y_range, color='white', linewidth=1)  # Flipped to align

    # Adjusted ellipses for the cropped region
    ax.add_patch(patches.Ellipse((0, 0), width=major_axis_R, height=minor_axis_R,
                                 angle=angle_R, edgecolor='cyan', facecolor='none',
                                 linewidth=1, linestyle="dotted"))
    ax.add_patch(patches.Ellipse((0,0), width=major_axis_s, height=minor_axis_s,
                                 angle=angle_s, edgecolor='red', facecolor='none',
                                 linewidth=1, linestyle="dotted"))

    # Add text labels near ellipses inside the image
    ax.text( + major_axis_R / 2, - minor_axis_R / 2, r"4 R", color="cyan", fontsize=10, weight="bold", ha='center')
    ax.text( - major_axis_R / 2,  - minor_axis_R / 2, r"4 $\sigma$", color="red", fontsize=10, weight="bold",
            ha='center')
    # Plot cropped image
    extent = [- (Cx-x_min*res), cropped_img_width_mm - (Cx-x_min*res), cropped_img_height_mm - (Cy-y_min*res),- (Cy-y_min*res)]  # Adjusting for mm scaling
    ax.imshow(cropped_image, cmap='inferno', origin='upper', extent=extent)
    # Set xlim and ylim to match image dimensions in mm
    ax.set_xlim([-cropped_img_width_mm/2, cropped_img_width_mm/2])
    ax.set_ylim([cropped_img_height_mm/2,-cropped_img_height_mm/2])  # Inverted to match image coordinates

    # Set axis labels in mm
    ax.set_xlabel("X [mm]", color="white", fontsize=12)
    ax.set_ylabel("Y [mm]", color="white", fontsize=12)

    # Set ticks and labels to white for visibility
    ax.tick_params(axis='both', colors='white', labelsize=10)
    # Define the number of ticks you want
    num_ticks_x = 6  # Increase for finer x-axis ticks
    num_ticks_y = 6  # Increase for finer y-axis ticks

    # Generate tick positions
    x_ticks = np.linspace(-cropped_img_width_mm / 2, cropped_img_width_mm / 2, num_ticks_x)
    y_ticks = np.linspace(cropped_img_height_mm / 2, -cropped_img_height_mm / 2, num_ticks_y)

    # Round to two decimal places
    x_ticks = np.round(x_ticks, 1)
    y_ticks = np.round(y_ticks, 1)

    # Ensure 0 is in the tick list
    if 0 not in x_ticks:
        x_ticks = np.sort(np.append(x_ticks, 0.0))
    if 0 not in y_ticks:
        y_ticks = np.sort(np.append(y_ticks, 0.0))

    # Apply ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set tick parameters
    ax.tick_params(axis='both', colors='white', labelsize=10)
    # Apply ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_frame_on(True)

    # Save the figure
    plt.savefig(f"{overlay_images_dir}/overlay_{index}.png", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def compute_beam_parameters(masked_image, threshold, res, compute_covariance=True):
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
        Sxy = -(Su ** 2 - Sv ** 2) / 4
    # Compute RMS sizes
    Rx, Ry, Rxy = calculate_rms_vectorized(temp_im, Cx, Cy, compute_covariance)

    angle = 0.5 * np.arctan2(2 * Rxy, Rx - Ry)
    return Cx, Cy, Sx, Sy, Sxy, Rx, Ry, Rxy, angle, res


def find_threshold_crossing(masked_image, threshold_images_dir, index, debug):
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
        Cx[j], Cy[j], Sx[j], Sy[j], _, Rx[j], Ry[j], _, _, _ = compute_beam_parameters(masked_image, bg, 1)

    print("Done processing")

    # Compute derivatives
    grad_Rx = np.gradient(Rx, thresholds)
    grad_Ry = np.gradient(Ry, thresholds)

    # Compute magnitudes
    magnitude = np.sqrt(np.abs(grad_Rx * grad_Ry))
    mean_threshold = np.mean(magnitude)

    # Find the first peak
    peaks, _ = find_peaks(magnitude)

    if len(peaks) > 0:
        max_idx = peaks[0]  # First detected peak
    else:
        max_idx = np.argmax(magnitude)  # Fallback to global maximum

    #print(f"Using max_idx = {max_idx}")

    # Find first crossing after the peak
    crossing_idx = np.where((magnitude[max_idx:] < mean_threshold))[0]
    if len(crossing_idx) > 0:
        crossing_idx = crossing_idx[0] + max_idx  # Adjust for slicing offset
        threshold_value = thresholds[crossing_idx]
        print(f"Threshold where RMS gradient crosses mean after peak: {threshold_value}")
    else:
        threshold_value = None
        print("No crossing found after maximum.")

    if debug == True:
        # Plot for visualization
        plt.plot(thresholds, np.sqrt(Sx * Sy) / np.max(np.sqrt(Sx * Sy)), label="Normalized |Sxy|")
        plt.plot(thresholds, np.sqrt(Rx * Ry) / np.max(np.sqrt(Rx * Ry)), label="Normalized |Rxy|")
        plt.plot(thresholds, magnitude, label="|dRx/dThresholds|")
        plt.axhline(mean_threshold, color='red', linestyle='--', label="Mean Threshold")
        if threshold_value is not None:
            plt.axvline(threshold_value, color='green', linestyle='--', label="Threshold Crossing")
        plt.legend()
        plt.xlabel("Threshold value")
        # Save the figure
        plt.savefig(f"{threshold_images_dir}/threshold_{index}.png")
        plt.close()

    return threshold_value
"""
# Function to filter the image 
def filter_image(image, sigma=2):
    return gaussian_filter(image, sigma=sigma)


# Function to calculate Sx, Sy, and correlation
def compute_statistics(image):
    x_projection = np.sum(image, axis=0)
    y_projection = np.sum(image, axis=1)
    x = np.arange(len(x_projection))
    y = np.arange(len(y_projection))

    Sx = np.std(x_projection)
    Sy = np.std(y_projection)
    correlation = np.corrcoef(x_projection, y_projection[:len(x_projection)])[0, 1]

    return Sx, Sy, correlation


# Function to plot projections
def plot_projections(image):
    x_projection = np.sum(image, axis=0)
    y_projection = np.sum(image, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(x_projection)
    ax1.set_title("X Projection")
    ax2.plot(y_projection)
    ax2.set_title("Y Projection")
    plt.show()


# Function to save results in a JSON file
def save_results(filename, Sx, Sy, correlation):
    results = {
        "Sx": Sx,
        "Sy": Sy,
        "Correlation": correlation
    }
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)"""