import numpy as np
import cv2 as cv
from image import pad_at_top_left

# img = np.arange(25).reshape((5, 5)) * 1.0
# img = img + 1

img = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]], np.float32)
print(img)
iy, ix = img.shape
ker = np.array([[1, 4, 2], [2, 0, -1], [3, 6, -2]])
print(ker)
# ker = np.array([[0, 1, 0], [-1, 2, -1], [0, 1, 0]], dtype=np.float32)
ky, kx = ker.shape

r = cv.filter2D(img, -1, ker, borderType=cv.BORDER_CONSTANT)
print("Convolution result:\n", r)

# img = cv.copyMakeBorder(img, 0, ky-1, 0, kx-1, cv.BORDER_CONSTANT, value=0)
img = pad_at_top_left(img, ky-1, kx-1)
print(img)
ker = np.rot90(ker, 2)
ker = cv.copyMakeBorder(ker, 0, iy-1, 0, ix-1, cv.BORDER_CONSTANT, value=0)
print(ker)

result = np.real(np.fft.ifft2(np.fft.fft2(ker) * np.fft.fft2(img)))
print("Multiplication result:\n", result)
print("Multiplication result after shift:\n", np.fft.fftshift(result))