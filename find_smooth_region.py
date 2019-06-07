import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def find_sd(img, table, x1, x2, y1, y2):
    sum2 = int(table[x1+1][y1+1] + table[x2][y2] - table[x1+1][y2] - table[x2][y1+1])
    average = float(sum2 / (x1 - x2 + 1) / (y1 - y2 + 1))
    block = img[x2:x1+1, y2:y1+1]
    # print(block)
    # print('/')
    block = block - average
    # print(block)
    block /= 255
    sd1 = float(0)
    for i in block:
        for j in i:
            sd1 = sd1 + float(j * j)
    sd1 = sd1 / (x1 - x2 + 1) / (y1 - y2 + 1)
    sd1 = float(sd1 ** 0.5)
    # print(block)
    # print('//')
    # print(sd1)
    return sd1


def find_smooth_region(img, kernel_w, kernel_h):
    """
    This function is to find the smooth region of a image, corresponding to the "Local prior p(L)" part in the paper.
    We build a table to store the sum of some boxes in the image, thus we can calculate the means fast.
    :param img:         the grayscale of the input image
    :param kernel_w:    kernel width
    :param kernel_h:    kernel height
    :return:            a black-and-white image which stores the smooth region
    """
    # first build a table for dynamic programming
    rows, cols = img.shape
    table = np.zeros((rows+1, cols+1))
    # print(table.shape)
    for i in range(rows):
        for j in range(cols):
            table[i+1][j+1] = table[i][j+1] + table[i+1][j] + img[i][j] - table[i][j]
    # print(table)
    kw = int((kernel_w - 1)/2)
    kh = int((kernel_h - 1)/2)
    g = np.zeros(img.shape)
    for i in range(rows):
        for j in range(cols):
            if find_sd(img, table, min(i+kw, rows-1), max(i-kw, 0), min(j+kh, cols-1), max(j-kh, 0)) < 0.02:
                g[i][j] = 255
            else:
                g[i][j] = 0

    cv.imshow('res', g)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return g


if __name__ == '__main__':
    """
        for function testing
    """
    img1 = cv.imread('./img/man.png', 0)

    # kernel_w1 = int(input('请输入卷积核的宽度'))  # 宽度和高度必须都为奇数
    # kernel_h1 = int(input('请输入卷积核的高度'))
    kernel_w1 = 11
    kernel_h1 = 11

    print(img1.shape)
    # print(img1)

    cv.imshow('image', img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

    find_smooth_region(img1, kernel_w1, kernel_h1)

    # print(img1)

