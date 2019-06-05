import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image import *
from update_psi import update_psi

kernel_size = (11, 11)
window_size = kernel_size
smooth_threshold = 5
# file_path = "./img/asc-s.jpg"
file_path = "./img/man.png"


if __name__ == '__main__':

    observed_image = load_image(file_path)

    # takes too long (~4min for 1900 * 1000 rgb images)
    smooth_mask = get_smooth_mask(observed_image, window_size, smooth_threshold)
    print('smooth_mask shape', smooth_mask.shape)
    cv.imwrite('smooth_mask.bmp', smooth_mask)
    smooth_img = cv.bitwise_and(observed_image, observed_image, mask=smooth_mask)
    plt.subplot(131); plt.imshow(smooth_img)

    kernel = initial_kernel(kernel_size)

    latent_image = observed_image

    psi = get_partial(latent_image)
    plt.subplot(132); plt.imshow(psi[0])
    plt.subplot(133); plt.imshow(psi[1])
    plt.show()

    psi = update_psi(psi, smooth_mask, observed_image, latent_image)
    plt.subplot(235); plt.imshow(psi[0])
    plt.subplot(236); plt.imshow(psi[1])
    plt.show()

    # gray_img = cv.cvtColor(latent_image, code=cv.COLOR_RGB2GRAY)
    # f = np.fft.fft2(gray_img)
    # f_shift = np.fft.fftshift(f)
    # magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    # plt.subplot(211); plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.show()


    # # optimize L and f
    # iter_n = 0
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


# cv.getDerivKernels()
# cv.getGaussianKernel()
# cv.filter2D()
