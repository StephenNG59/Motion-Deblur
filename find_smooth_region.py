import numpy as np
import cv2 as cv
import time
from matplotlib import pyplot as plt


def find_sd(img, table, channels, x1, x2, y1, y2):
    sum1 = 0
    for c in range(channels):
        sum2 = int(table[x1+1][y1+1][c] + table[x2][y2][c] - table[x1+1][y2][c] - table[x2][y1+1][c])
        average = float(sum2 / (x1 - x2 + 1) / (y1 - y2 + 1))
        block = img[x2:x1+1, y2:y1+1, c]
        # print(block)
        # print('/')
        block = block - average
        # print(block)
        # block /= 255
        sd1 = float(0)
        for i in block:
            for j in i:
                sd1 = sd1 + float(j * j)
        sd1 = sd1 / (x1 - x2 + 1) / (y1 - y2 + 1)
        sd1 = float(sd1 ** 0.5)
        # print(block)
        # print('//')
        # print(sd1)
        sum1 += sd1
    return sum1


def find_smooth_region(img, ker_shape, threshold):
    """
    This function is to find the smooth region of a image, corresponding to the "Local prior p(L)" part in the paper.
    We build a table to store the sum of some boxes in the image, thus we can calculate the means fast.
    :param img:         the grayscale of the input image
    :param ker_shape:   shape of slide window, which is the same as kernel shape
    :param threshold:   the threshold used for the image sd, we set it to 0.02 here
    :return:            a black-and-white image which stores the smooth region
    """
    # first build a table for dynamic programming
    rows, cols, channels = img.shape
    table = np.zeros((rows+1, cols+1, channels))
    # print(table.shape)
    for c in range(channels):
        for i in range(rows):
            for j in range(cols):
                table[i+1][j+1][c] = table[i][j+1][c] + table[i+1][j][c] + img[i][j][c] - table[i][j][c]
    # print(table)
    kw = int((ker_shape[0] - 1) / 2)
    kh = int((ker_shape[1] - 1) / 2)
    # kw = int((kernel_w - 1)/2)
    # kh = int((kernel_h - 1)/2)
    smooth_mask = np.zeros(img.shape)
    for i in range(rows):
        for j in range(cols):
            if find_sd(img, table, channels, min(i+kw, rows-1), max(i-kw, 0), min(j+kh, cols-1), max(j-kh, 0))\
                    < threshold:
                smooth_mask[i][j] = 255
            else:
                smooth_mask[i][j] = 0

    # cv.imshow('res', smooth_mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return smooth_mask


if __name__ == '__main__':
    """
        for function testing
    """
    img1 = cv.imread('./img/man.png')

    # kernel_w1 = int(input('请输入卷积核的宽度'))  # 宽度和高度必须都为奇数
    # kernel_h1 = int(input('请输入卷积核的高度'))

    print(img1.shape)
    # print(img1)

    cv.imshow('image', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

    find_smooth_region(img1, (5, 5), 5)

    # print(img1)

