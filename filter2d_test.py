import cv2 as cv
import numpy as np

mat1 = np.array([[1, 2, 4], [8, 16, 32], [64, 128, 256]], np.float32)
img = cv.merge((mat1, mat1, mat1))
print(mat1)

d0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
dx = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
dy = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], np.float32)
dxx = np.zeros((5, 5))
dxx[2][2] = -0.5
dxx[2][0] = 0.25
dxx[2][4] = 0.25

kernel = np.array([[1, 1, 0], [0, 0, 0], [0, 0, -1]])
print(dxx)

res = cv.filter2D(mat1, -1, dx, borderType=cv.BORDER_REPLICATE)
print(res)

res = cv.filter2D(res, -1, dx, borderType=cv.BORDER_REPLICATE)
print(res)

res = cv.filter2D(mat1, -1, dxx, borderType=cv.BORDER_REPLICATE)
print(res)
