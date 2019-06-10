import numpy as np
import cv2 as cv

img = np.arange(25).reshape((5, 5)) * 1.0
ix = img.shape[0]
ker = np.array([[0, 1, 0], [0, 2, 1], [1, 0, 0]], np.float32)
kx = ker.shape[0]

print("Convolution result:", cv.filter2D(img, -1, ker, borderType=cv.BORDER_REPLICATE))

img_pad = cv.copyMakeBorder(img, 0, kx-1, 0, kx-1, cv.BORDER_CONSTANT, value=0)
ker_pad = cv.copyMakeBorder(ker, 0, ix-1, 0, ix-1, cv.BORDER_CONSTANT, value=0)

print("Multiplication result:", np.real(np.fft.ifft2(np.fft.fft2(img_pad) * np.fft.fft2(ker_pad))))
