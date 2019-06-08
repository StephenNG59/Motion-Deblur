import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
import cv2 as cv

mat1 = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float)
F_mat1 = np.fft.fft2(mat1)
print(mat1)
# mat2 = np.array([[1, 2, 4], [8, 16, 32], [64, 128, 256]], dtype=float)
mat2 = np.array([[1, 2, 4, 8, 16], [32, 64, 128, 256, 512], [3, 9, 27, 81, 243], [5, 25, 125, 625, 3125],
                 [-1, -2, -4, -8, -16]], dtype=float)
F_mat2 = np.fft.fft2(mat2)
print(mat2)

e_w_p = F_mat1 * F_mat2

i_e_w_p = np.fft.ifft2(e_w_p)
print(i_e_w_p)

dst = cv.filter2D(mat2, -1, mat1)
print(dst)
