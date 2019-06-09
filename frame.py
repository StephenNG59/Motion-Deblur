import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image import *
from update_psi import *
from Latent import *
import time
from vars import *


if __name__ == '__main__':

    observed_image = load_image(file_path)

    # get smooth region mask
    print("1. Getting smooth region...")
    t1 = time.time()
    smooth_mask = get_smooth_mask(observed_image, window_size, smooth_threshold)
    mask_01 = smooth_mask / 255.0
    print("- 1.finished time {0}".format(time.time() - t1))
    plt.subplot(111); plt.imshow(smooth_mask[:, :, 0], cmap='gray'); plt.show()

    kernel = initial_kernel(kernel_size)

    latent = Latent(observed_image, kernel.shape)

    psi = get_partial(latent.latent)

    iter_n = 0
    flag1 = True
    while flag1:
        iter_n = iter_n + 1
        flag2 = True
        while flag2:
            psi_origin = psi
            psi_updated = update_psi(psi, mask_01, observed_image, latent.latent, Gamma, Lambda1, Lambda2)
            # psi_updated = (psi_updated[0] * 1.05, psi_updated[1] * 1.05)  # for test
            psi_diff = get_psi_diff(psi_updated, psi_origin)
            latent_diff = latent.update_l(kernel, psi_updated)
            print("latent_diff = {0:.5f}, psi_diff = {1:.10f}".format(latent_diff, psi_diff))
            if latent_diff < LatentThreshold and psi_diff < PsiThreshold:
                flag2 = False
            psi = psi_updated
            plt.subplot(111); plt.imshow(latent.latent); plt.show()
        flag1 = False    # for test
        # update f
        # updated_f = update_f()
        # if f_diff(updated_f, f) < f_threshold or iters >= iters_max:
            # flag1 = False
        # f = updated_f



