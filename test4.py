# 直接把blur图片在频域中除以kernel的fft，得到的是一堆rgb灯效……为什么呢

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from image import pad_at_top_left

img = cv.imread("./img/test4.jpg"); img = img / 255.0; iy, ix, _ = img.shape
cv.imshow("Input", img)
cv.waitKey(0)
x = 5; kernel = np.ones((x, x)); kernel[x//2][x//2] = 1.0; ky, kx = kernel.shape

img_pad = pad_at_top_left(img, ky-1, kx-1); r, g, b = cv.split(img_pad)
ker_pad = pad_at_top_left(kernel, iy-1, ix-1)
rr, gg, bb = \
    np.real(np.fft.ifft2(np.fft.fft2(r) / np.fft.fft2(ker_pad))), \
    np.real(np.fft.ifft2(np.fft.fft2(g) / np.fft.fft2(ker_pad))), \
    np.real(np.fft.ifft2(np.fft.fft2(b) / np.fft.fft2(ker_pad)))
plt.subplot(111); plt.imshow(rr); plt.show()
out = cv.merge([rr, gg, bb])
cv.imshow("Result", out)
cv.waitKey(0)