### functions.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.ndimage import median_filter, rotate
from matplotlib.widgets import RectangleSelector

# Function to open H5 file and read images
def get_images(file_path):
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        res = f['images'].attrs['resolution']
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

# Function to let user select a draggable ROI
def select_roi(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
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
        return image[y1:y2, x1:x2]
    else:
        return None  # In case selection wasn't made

def denoise_median_arrayed(data, size):
    # Apply median filter with a specified size (window size)
    filtered_data = np.apply_along_axis(lambda x: median_filter(x, size=2), axis=1, arr=data)
    filtered_data = median_filter(filtered_data, size=size)
    return filtered_data

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