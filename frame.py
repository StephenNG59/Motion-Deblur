import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image import *
from update_psi import *
from update_f import *
from Latent import *
import time
from vars import *
from find_smooth_region import *


get_smooth = False
operating_f = False
smooth_npy = "./img/smooth.npy"

if __name__ == '__main__':

    observed_image = load_image(file_path)
    kernel = initial_kernel()
    shape = observed_image.shape
    shape_x, shape_y, _ = shape
    x1 = shape_x // 2 - 40
    x2 = shape_x // 2 + 40
    y1 = shape_y // 2 - 40
    y2 = shape_y // 2 + 40

    # get smooth region mask
    print("1. Getting smooth region...")
    t1 = time.time()
    if get_smooth:
        smooth_mask = find_smooth_region(observed_image, kernel.shape, smooth_threshold)
        print("111111")
        cv.imshow('res', smooth_mask)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # np.save(smooth_npy, smooth_mask)
        # smooth_mask_saved = cv.merge((smooth_mask, smooth_mask, smooth_mask))
    else:
        # smooth_mask = np.load(smooth_npy)
        smooth_mask = cv.imread('img/smooth_mask/mask.png', 0)
        # smooth_mask = smooth_mask_saved[:, :, 0]
    mask_01 = smooth_mask / 255.0
    print("- 1.finished time {0}".format(time.time() - t1))
    plt.subplot(111); plt.imshow(smooth_mask[:, :], cmap='gray'); plt.show()

    latent = Latent(observed_image, kernel.shape)
    cv.imwrite("./img/test/#1.jpg", 255 * latent.latent)

    psi = get_partial(latent.latent)
    cv.imwrite("./img/test/#2-x.jpg", (psi[0] + 255 / 2.0))

    iter_1 = 0
    flag1 = True
    updated_f = np.eye(27, 27)
    updated_f /= np.sum(updated_f)
    updated_f = kernel
    while flag1:
        iter_1 = iter_1 + 1
        iter_2 = 0
        flag2 = True
        while flag2:
            iter_2 = iter_2 + 1
            psi_origin = psi
            psi_updated = update_psi(psi, mask_01, observed_image, latent.latent, Gamma, Lambda1, Lambda2)
            # print(psi_updated)
            # psi_updated = (psi_updated[0] * 1.05, psi_updated[1] * 1.05)  # for test
            #psi_diff = get_psi_diff(psi_updated, psi_origin)
            psi_diff = 0
            latent_diff = latent.update_l(updated_f, psi_updated)
            print("latent_diff = {0:.5f}, psi_diff = {1:.10f}".format(latent_diff, psi_diff))
            if latent_diff < LatentThreshold and psi_diff < PsiThreshold:
                flag2 = False
            psi = psi_updated
            psi_x, psi_y = psi
            cv.imwrite('img/dx/dx_psi.png', psi_x * 255)
            cv.imwrite('img/dx/dy_psi.png', psi_y * 255)
            cv.imwrite("./img/out/out-" + str(iter_2) + ".jpg", latent.latent * 255)
            for tmp1 in latent.latent:
                for tmp2 in tmp1:
                    for para in tmp2:
                        if para > 1:
                            para = 1
                        elif para < 0:
                            para = 0

        # flag1 = False    # for testing
        # --------------------- update f ------------------------
        if operating_f:
            f = updated_f
            updated_f = update_f(latent.latent[x1:x2, y1:y2], observed_image[x1:x2, y1:y2], updated_f)
            f_diff_val = f_diff(updated_f, f)
            print("f_diff = {0:.5f}".format(f_diff_val))
            if f_diff_val < f_threshold:
                flag1 = False
        else:
            flag1 = False
        # f = updated_f

    lat = 255 * latent.latent
    cv.imwrite("./img/out/out.jpg", lat)
    cv.imwrite('img/kernel/kernel.png', updated_f * 255)
