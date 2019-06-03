import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt


def get_smooth_region(src, window_size, threshold):
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

    smooth_region = np.zeros((img_h, img_w), dtype=np.int8)
    for i in range(img_h):
        for j in range(img_w):
            # print(sd[i][j])
            if sd[i][j] < threshold:
                smooth_region[i][j] = 1

    return smooth_region
