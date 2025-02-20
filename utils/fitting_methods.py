import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor

from .batch_minimization import gen_candidates_scipy

logger = logging.getLogger(__name__)


def gaussian_linear_background(x, amp, mu, sigma, offset=0):
    """Gaussian plus linear background fn"""
    return amp * np.exp(-((x - mu) ** 2) / 2 / sigma**2) + offset


class GaussianLeastSquares:
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


def plot_fit(x, y, para_x):
    """
    Plot  beamsize fit in x or y direction
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, y, label="data")
    ax.plot(
        x,
        gaussian_linear_background(x, *para_x),
        "r-",
        label=f"fit: amp={para_x[0]:.2f}, centroid={para_x[1]:.2f}, sigma="
        f"{para_x[2]:.2f}, offset={para_x[3]:.3f}",
    )
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Counts")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()

    return fig, ax
