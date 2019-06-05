import numpy as np
import cv2 as cv


def load_image(file_path):
    im = cv.imread(file_path, cv.IMREAD_UNCHANGED)
    return im


def initial_kernel(size):

    # currently only supports square
    assert size[0] == size[1], "error: kernel size is not square"

    kernel = np.identity(size[0])
    # kernel = np.eye(size[0])

    return kernel


def get_partial(image):
    # psi = cv.spatialGradient(latent_image, ksize=3, borderType=cv.BORDER_DEFAULT)
    # d_depth = cv.CV_16S
    par_x = cv.Sobel(image, -1, dx=1, dy=0, ksize=3, borderType=cv.BORDER_DEFAULT)
    par_y = cv.Sobel(image, -1, dx=0, dy=1, ksize=3, borderType=cv.BORDER_DEFAULT)
    par_x = cv.convertScaleAbs(par_x)
    par_y = cv.convertScaleAbs(par_y)
    return par_x, par_y


def get_smooth_mask(src, window_size, threshold):
    """
    Compute smooth region based on threshold.
    :param src: source image [width of image, height of image, channels]
    :param window_size: tuple object - [height of window, width of window]
    :param threshold: threshold of standard deviation
    :return:
        smooth_region: a mask of the smooth region, same shape as src
    """

    img_h, img_w, img_c = src.shape
    win_h, win_w = window_size
    assert win_w % 2 == 1 & win_h % 2 == 1, "window size must be odd number"
    win_half_w, win_half_h = win_w // 2, win_h // 2

    # padding
    src_pad = cv.copyMakeBorder(src, win_half_h, win_half_h, win_half_w, win_half_w, cv.BORDER_REFLECT_101)

    # np.std()

    sd = np.zeros(src.shape)        # [h, w, c]
    for i in range(img_h):
        y = i + win_half_h
        for j in range(img_w):
            x = j + win_half_w
            window = src_pad[i:y+win_half_h, j:x+win_half_w, :]
            for k in range(img_c):
                sd[i][j][k] = np.std(window[:, :, k])
    sd = np.sum(sd, axis=2)         # add up standard deviation of all channels

    # smooth_region = cv.inRange()

    smooth_region = np.zeros((img_h, img_w, 1), np.uint8)
    for i in range(img_h):
        for j in range(img_w):
            # print(sd[i][j])
            if sd[i][j] < threshold:
                smooth_region[i][j] = 0xff

    return smooth_region
