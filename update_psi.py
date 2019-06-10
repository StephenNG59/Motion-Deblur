import numpy as np
from energy import get_cof
from image import get_partial
from vars import *


def update_psi(Psi, Mask, Observed_image, Latent_image, gamma=Gamma, lambda1=Lambda1, lambda2=Lambda2):
    """
    Update Psi.
    :param Psi: the substitution variable which gradually approaches the derivative of latent image
    :param Mask: mask of smooth region
    :param Observed_image:
    :param Latent_image:
    :param gamma: weight of substitution variable, should iteratively increases until sufficiently large
    :param lambda1: adjustable
    :param lambda2: adjustable
    :return:
        updated psi
    """

    Psi_x, Psi_y = Psi
    height, width, n_channel = Observed_image.shape
    d_Observed = get_partial(Observed_image)
    d_Latent = get_partial(Latent_image)

    # lambda1 * |psi| + lambda2 * mask * (psi - d(observed))^2 + gamma * (psi - d(latent))^2
    h = height // 10
    for i in range(height):
        if i % h == 0:
            print("Updating psi - progress {}%".format(i // h * 10))
        for j in range(width):
            for k in range(n_channel):
                psi_x, psi_y = Psi_x[i][j][k], Psi_y[i][j][k]
                mask = Mask[i][j][0]
                dx_obs, dx_lat = d_Observed[0][i][j][k], d_Latent[0][i][j][k]
                dy_obs, dy_lat = d_Observed[1][i][j][k], d_Latent[1][i][j][k]
                Psi_x[i][j][k] = update_single_psi(psi_x, mask, dx_obs, dx_lat, gamma, lambda1, lambda2)
                Psi_y[i][j][k] = update_single_psi(psi_y, mask, dy_obs, dy_lat, gamma, lambda1, lambda2)

    return Psi_x, Psi_y


def update_single_psi(psi, mask, d_observed, d_latent, gamma, lambda1, lambda2):
    k, a, b, lt = get_cof()

    psi1 = (lambda1 * k + 2 * lambda2 * mask * d_observed + 2 * gamma * d_latent) / (2 * (lambda2 * mask + gamma))
    if -lt <= psi1 <= 0:
        pass
    elif psi1 > 0:
        psi1 = 0
    elif psi1 < -lt:
        psi1 = -lt
    e1 = (-k * lambda1 * psi1) + lambda2 * mask * (psi1 - d_observed)**2 + gamma * (psi1 - d_latent)**2

    psi2 = (-lambda1 * k + 2 * lambda2 * mask * d_observed + 2 * gamma * d_latent) / (2 * (lambda2 * mask + gamma))
    if 0 <= psi2 <= lt:
        pass
    elif psi2 < 0:
        psi2 = 0
    elif psi2 > lt:
        psi2 = lt
    e2 = (k * lambda1 * psi2) + lambda2 * mask * (psi2 - d_observed) ** 2 + gamma * (psi2 - d_latent) ** 2

    psi3 = (lambda2 * mask * d_observed + gamma * d_latent) / (lambda1 * a + lambda2 * mask + gamma)
    if -lt < psi3 < lt:
        psi3 = lt
    e3 = lambda1 * (a*psi3*psi3 + b) + lambda2 * mask * (psi3 - d_observed) ** 2 + gamma * (psi3 - d_latent) ** 2

    e4 = (k * lambda1 * lt) + lambda2 * mask * (-lt - d_observed)**2 + gamma * (-lt - d_latent)**2

    e_min = min(e1, e2, e3, e4)

    if e_min == e1:
        psi_star = psi1
    elif e_min == e2:
        psi_star = psi2
    elif e_min == e3:
        psi_star = psi3
    elif e_min == e4:
        psi_star = -lt

    return psi_star


def get_psi_diff(psi_updated, psi):
    psi_diff_x = psi_updated[0] - psi[0]
    psi_diff_y = psi_updated[0] - psi[0]

    diff_r = max(np.linalg.norm(psi_diff_x[:, :, 0], ord=2), np.linalg.norm(psi_diff_y[:, :, 0], ord=2))
    diff_g = max(np.linalg.norm(psi_diff_x[:, :, 1], ord=2), np.linalg.norm(psi_diff_y[:, :, 1], ord=2))
    diff_b = max(np.linalg.norm(psi_diff_x[:, :, 2], ord=2), np.linalg.norm(psi_diff_y[:, :, 2], ord=2))

    diff = max(diff_r, diff_g, diff_b)
    return diff
