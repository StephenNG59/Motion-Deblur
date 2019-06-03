import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image import load_image, get_partial
from get_smooth_region import get_smooth_region
from kernel import initial_kernel

kernel_size = (11, 11)
window_size = kernel_size
smooth_threshold = 5
file_path = "./img/asc.jpg"


if __name__ == '__main__':
    observed_image = load_image(file_path)
    # takes too long (~4min for 1900 * 1000 rgb images)
    # smooth_region = get_smooth_region(observed_image, window_size, smooth_threshold)
    kernel = initial_kernel(kernel_size)

    latent_image = observed_image

    psi = get_partial(latent_image)
    # psi = cv.spatialGradient(latent_image, ksize=3, borderType=cv.BORDER_DEFAULT)

    iter_n = 0

    plt.subplot(221); plt.imshow(psi[0])
    plt.subplot(222); plt.imshow(psi[1])
    plt.show()



    # # optimize L and f
    # flag1 = True
    # while flag1:
    #     iter_n = iter_n + 1
    #     # optimize L
    #     flag2 = True
    #     while flag2:
    #         updated_psi = update_psi()
    #         updated_latent_image = compute_latent_image()
    #         if image_diff(updated_latent_image, latent_image) < latent_threshold and psi_diff(updated_psi, psi) < psi_threshold:
    #             flag2 = False
    #         psi = updated_psi
    #         latent_image = updated_latent_image
    #     # update f
    #     updated_f = update_f()
    #     if f_diff(updated_f, f) < f_threshold or iters >= iters_max:
    #         flag1 = False
    #     f = updated_f
    #
    # output(latent_image, f)