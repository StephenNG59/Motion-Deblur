# import numpy as np
from energy import get_cof
from image import get_partial


def update_psi(Psi, Mask, Observed_image, Latent_image, gamma=1, lambda1=0.002, lambda2=10):
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

    if psi < -lt or psi > lt:
        psi_ = (lambda2 * mask * d_observed + gamma * d_latent) / (lambda1 * a + lambda2 * mask + gamma)
        if psi_ < -lt or psi_ > lt:
            psi_star = psi_
        elif lt >= psi_ >= 0:
            psi_star = lt
        else:               # -lt <= psi_ < 0
            psi_star = lt

    elif -lt <= psi < 0:
        psi_ = (lambda1 * k + 2 * lambda2 * mask * d_observed + 2 * gamma * d_latent) / (2 * (lambda2 * mask + gamma))
        if -lt <= psi_ < 0:
            psi_star = psi_
        elif psi_ >= 0:
            psi_star = 0
        else:               # psi_ < -lt
            psi_star = -lt

    else:                   # 0 <= psi <= lt
        psi_ = (-lambda1 * k + 2 * lambda2 * mask * d_observed + 2 * gamma * d_latent) / (2 * (lambda2 * mask + gamma))
        if 0 <= psi_ <= lt:
            psi_star = psi_
        elif psi_ < 0:
            psi_star = 0
        else:               # psi_ > lt
            psi_star = lt

    return psi_star
