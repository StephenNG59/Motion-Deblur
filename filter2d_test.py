import cv2 as cv
import numpy as np

mat1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(mat1)

kernel = np.array([[1, 1, 0], [0, 0, 0], [0, 0, -1]])
print(kernel)

res =