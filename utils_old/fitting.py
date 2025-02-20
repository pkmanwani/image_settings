import numpy as np

from utils import fit_gaussian_linear_background

def fit_image(img):
    x_projection = np.sum(img, axis=0)
    y_projection = np.sum(img, axis=1)

    # subtract min value from projections
    x_projection = x_projection - x_projection[:10].min()
    y_projection = y_projection - y_projection[:10].min()

    para_x = fit_gaussian_linear_background(x_projection)
    para_y = fit_gaussian_linear_background(x_projection)
