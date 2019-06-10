# import numpy as np
# import cv2 as cv
# from image import pad_at_top_left
#
# img = cv.imread("./img/test5.jpg"); img = img / 255.0; iy, ix, _ = img.shape
# x = 5; kernel = np.ones((x, x)); kernel[x//2][x//2] = 1.0; ky, kx = kernel.shape
#
# img_pad = pad_at_top_left(img, ky-1, kx-1); r, g, b = cv.split(img_pad)
# ker_pad = pad_at_top_left(kernel, iy-1, ix-1)

import numpy as np
import cv2 as cv

image = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]], np.float32)
image = cv.copyMakeBorder(image, 0, 2, 0, 1, cv.BORDER_CONSTANT, value=0)
print(image)

# kernel = np.array([[1, 4, 0, 0, 0], [2, 5, 0, 0, 0], [3, 6, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                   # [0, 0, 0, 0, 0]], np.float32)
kernel = np.array([[1, 4], [2, 5], [3, 6]], np.float32)
kernel = cv.copyMakeBorder(kernel, 0, 3, 0, 3, cv.BORDER_CONSTANT, value=0)
print(kernel)

f = np.fft.ifft2(np.fft.fft2(image)*np.fft.fft2(kernel))
print(f)