import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
import cv2 as cv
from scipy import signal

img = np.array([[1, 2, 4, 8, 16], [32, 64, 128, 256, 512], [3, 9, 27, 81, 243], [5, 25, 125, 625, 3125],
                [-1, -2, -4, -8, -16]], dtype=float)

kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)

dst = signal.convolve2d(img, kernel, mode='full', boundary='fill', fillvalue=0)
print(dst)

img1 = np.array([[1, 2, 4, 8, 16, 0, 0], [32, 64, 128, 256, 512, 0, 0], [3, 9, 27, 81, 243, 0, 0],
                 [5, 25, 125, 625, 3125, 0, 0], [-1, -2, -4, -8, -16, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]], dtype=float)

kernel1 = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=float)

F_image = np.fft.fft2(img1)
F_kernel = np.fft.fft2(kernel1)
ew = F_kernel*F_image
iew = np.fft.ifft2(ew)
print(iew)
