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
    Xnom = np.sum(image * dx_grid**2)
    Ynom = np.sum(image * dy_grid**2)

    # RMS size is the square root of variance
    Rx = np.sqrt(Xnom / total_intensity)
    Ry = np.sqrt(Ynom / total_intensity)

    return Cx,Cy,Rx,Ry

#
def gaussian_linear_background(x, amp, mu, sigma, offset=0):
    """Gaussian plus linear background fn"""
    return amp * np.exp(-((x - mu) ** 2) / 2 / sigma**2) + offset


class GaussianLeastSquares:
    """
        Calculate the centroid (Cx, Cy) and sigmas assuming Gaussian (Sx, Sy) of an image.

        Args:
            image (ndarray): The input 2D image (intensity array).

        Returns:
            Cx (float): The x-coordinate of the centroid.
            Cy (float): The y-coordinate of the centroid.
            Sx (float): The RMS size in the x-direction.
            Sy (float): The RMS size in the y-direction.
        """
    def __init__(self, train_x: Tensor, train_y: Tensor, pk_loc):
        self.train_x = train_x
        self.train_y = train_y
        self.pk_loc = pk_loc

    def forward(self, X):
        amp = X[..., 0].unsqueeze(-1)
        mu = X[..., 1].unsqueeze(-1)
        sigma = X[..., 2].unsqueeze(-1)
        offset = X[..., 3].unsqueeze(-1)
        train_x = self.train_x.repeat(*X.shape[:-1], 1)
        train_y = self.train_y.repeat(*X.shape[:-1], 1)
        pred = amp * torch.exp(-((train_x - mu) ** 2) / 2 / sigma**2) + offset
        neg_mse = -torch.sum((pred - train_y) ** 2, dim=-1).sqrt() / len(train_y)
        neg_log_prior_loss = (
            -0.01 * (amp.squeeze() - 1.0) ** 2
            - 0.01**2 * (mu.squeeze() - self.pk_loc) ** 2
        )
        # print(
        #    float(torch.mean(neg_mse)),
        #    float(torch.mean(neg_log_prior_loss))
        # )
        loss = neg_mse + neg_log_prior_loss

        return loss


def fit_gaussian_linear_background(y, inital_guess=None, visualize=True, n_restarts=1):
    """
    Takes a function y and inputs and fits and Gaussian with
    linear bg to it. Returns the best fit estimates of the parameters
    amp, mu, sigma and their associated 1sig error
    """

    # threshold off mean value on the edge of the domain
    thresholded_y = y - np.mean(y[-10:])
    thresholded_y = np.where(thresholded_y > 0, thresholded_y, 0)

    # normalize amplitude
    normed_y = thresholded_y / max(thresholded_y)

    x = np.arange(normed_y.shape[0])
    width = normed_y.shape[0]
    inital_guess = inital_guess or {}
    sigma_min = 2.0

    # specify initial guesses if not provided in initial_guess
    smoothed_y = np.clip(gaussian_filter(normed_y, 3), 0, np.inf)

    pk_value = np.max(smoothed_y)
    pk_loc = np.argmax(smoothed_y)

    offset = inital_guess.pop("offset", np.mean(normed_y[-10:]))
    amplitude = inital_guess.pop("amplitude", smoothed_y.max() - offset)
    # slope = inital_guess.pop("slope", 0)

    # use weighted mean and rms to guess
    center = inital_guess.pop("mu", pk_loc)
    sigma = inital_guess.pop("sigma", normed_y.shape[0] / 2)

    para0 = torch.tensor([amplitude, center, sigma, offset])

    # generate points +/- 50 percent
    rand_para0 = torch.rand((n_restarts, 4)) - 0.5
    rand_para0[..., 0] = (rand_para0[..., 0] + 1.0) * amplitude
    rand_para0[..., 1] = (rand_para0[..., 1] + 1.0) * center
    rand_para0[..., 2] = (rand_para0[..., 2] + 1.0) * sigma
    rand_para0[..., 3] = rand_para0[..., 3] * 200 + offset

    para0 = torch.vstack((para0, rand_para0))

    bounds = torch.tensor(
        (
            (pk_value / 2.0, max(center - width / 4, 0), sigma_min, 0.0),
            (pk_value * 1.5, min(center + width / 4, width), width, 1.2),
        )
    )

    # clip on bounds
    para0 = torch.clip(para0, bounds[0], bounds[1])

    # create LSQ model
    model = GaussianLeastSquares(
        torch.tensor(x), torch.tensor(normed_y), torch.tensor(pk_loc)
    )
    smoothed_model = GaussianLeastSquares(
        torch.tensor(x), torch.tensor(smoothed_y), torch.tensor(pk_loc)
    )

    # fit smoothed model to get better initial points
    scandidates, svalues = gen_candidates_scipy(
        para0,
        smoothed_model.forward,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        options={"maxiter": 50},
    )

    # fit regular model to refine
    candidates, values = gen_candidates_scipy(
        scandidates,
        smoothed_model.forward,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        options={"maxiter": 50},
    )

    # in some cases the fit will return a sigma value of 2.0
    # or an amplitude that is within the noise
    # drop these from candidates
    # print(candidates)
    indiv_condition = torch.stack(
        (
            candidates[:, -2] > sigma_min * 1.1,
            candidates[:, -2] < width / 1.5,
            candidates[:, 0] > 0.1,
        )
    )
    # print(indiv_condition)

    condition = torch.all(indiv_condition, dim=0)
    # print(condition)
    valid_candidates = candidates[condition]
    valid_values = values[condition]

    if len(valid_candidates) > 0:
        # get best valid from restarts
        candidate = valid_candidates[torch.argmax(valid_values)].detach().numpy()

        if visualize:
            plot_fit(x, normed_y, candidate)

    else:
        # if no fits were successful return nans
        bad_candidate = candidates[torch.argmax(values)].detach().numpy()
        if visualize:
            fig, ax = plot_fit(x, normed_y, bad_candidate)
            ax.set_title("bad fit")

        candidate = [np.nan] * 4

    return candidate
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