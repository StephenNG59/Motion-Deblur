import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image import *
from update_psi import update_psi
import time

kernel_size = (11, 11)
window_size = kernel_size
# smooth_threshold = 0.05
smooth_threshold = 5
# file_path = "./img/asc-s.jpg"
file_path = "./img/man-m.png"
mask_path = "./img/mask-m.bmp"
mask_np = "./img/mask.npy"
psi_updated_path = "./img/psi_updated.npy"


if __name__ == '__main__':

    observed_image = load_image(file_path)

    # get smooth region mask
    # t1 = time.time()
    # smooth_mask = get_smooth_mask(observed_image, window_size, smooth_threshold)
    # print('Get smooth mask time:', time.time() - t1)
    # np.save(mask_np, smooth_mask)
    # cv.imwrite(mask_path, smooth_mask)
    smooth_mask = np.load(mask_np)
    mask_01 = smooth_mask / 255.0
    # smooth_img = cv.bitwise_and(observed_image, observed_image, mask=smooth_mask)     # not suitable for float type
    # smooth_img = observed_image * smooth_mask / 255.0
    # plt.subplot(131); plt.imshow(smooth_img)

    kernel = initial_kernel(kernel_size)

    latent_image = observed_image

    # initial psi
    t1 = time.time()
    psi = get_partial(latent_image)
    print('Get partial time:', time.time() - t1)
    # region plot partial of image
    plt.subplot(132); plt.imshow(psi[0] / 510.0 + 0.5)
    plt.subplot(133); plt.imshow(psi[1] / 510.0 + 0.5)
    plt.show()
    # endregion

    # update psi
    t1 = time.time()
    psi_updated = update_psi(psi, mask_01, observed_image, latent_image)
    np.save(psi_updated_path, psi_updated)
    # psi_updated = np.load(psi_updated_path)
    print('Update psi time:', time.time() - t1)
    # region plot psi updating result
    # plt.subplot(321); plt.imshow(psi[0])
    # plt.subplot(322); plt.imshow(psi[1])
    # plt.subplot(323); plt.imshow(psi_updated[0])
    # plt.subplot(324); plt.imshow(psi_updated[1])
    # plt.subplot(325); plt.imshow(psi_updated[0] - psi[0])
    # plt.subplot(326); plt.imshow(psi_updated[1] - psi[1])
    # plt.show()
    # print(psi_updated[0] - psi[0])
    # endregion
    # psi_diff =

    # update latent image

    gray_img = cv.cvtColor(latent_image, code=cv.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray_img)
    dx = np.array([[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]], dtype=np.float32)
    # print(np.fft.fft2(dx))
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
