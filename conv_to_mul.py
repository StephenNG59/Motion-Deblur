import numpy as np
from image import initial_kernel
import cv2 as cv


def conv_to_mul(img, ker):
    result = np.zeros((img.size, ker.size))
    img_x, img_y, _ = img.shape
    ker_x, ker_y = ker.shape

    img_pad = cv.copyMakeBorder(img, ker_y//2, ker_y//2, ker_x//2, ker_x//2, cv.BORDER_CONSTANT)
    b, g, r = cv.split(img_pad)

    for i in range(img_x):
        for j in range(img_y):
            tmp = b[i:i+ker_x, j:j+ker_y]
            line = np.reshape(tmp, (1, -1))
            result[j + i * img_y, :] = line

    for i in range(img_x):
        for j in range(img_y):
            tmp = g[i:i + ker_x, j:j + ker_y]
            line = np.reshape(tmp, (1, -1))
            result[img_x * img_y + j + i * img_y, :] = line

    for i in range(img_x):
        for j in range(img_y):
            tmp = r[i:i+ker_x, j:j+ker_y]
            line = np.reshape(tmp, (1, -1))
            result[2 * img_x * img_y + j + i * img_y, :] = line

    return result


if __name__ == '__main__':
    ker = initial_kernel()
    matrix = np.eye(4, 3)
    img = cv.merge((matrix, matrix, matrix))
    res = conv_to_mul(img, ker)

    print(matrix)
    print(res)
