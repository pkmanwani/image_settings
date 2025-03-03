### functions.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.ndimage import median_filter, rotate
from matplotlib.widgets import RectangleSelector
import cv2
# Function to open H5 file and read images
def get_images(file_path):
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        res = f['images'].attrs['resolution']
        print(res)
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
    print("Original shape:", image.shape)

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