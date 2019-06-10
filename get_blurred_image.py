import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import Latent

img = cv.imread('img//girl2.png')
# cv.imshow('girl', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

k_w = 29
k_h = 5
kernel = np.zeros((k_h, k_w))
for i in range(k_w//2-10, k_w//2+11):
    kernel[k_h//2][i] = 0.05
kernel[k_h//2][k_w//2] = 0.1

print(kernel)

img_blurred = cv.filter2D(img, -1, kernel)
cv.imshow('girl', img_blurred)
cv.waitKey(0)
cv.destroyAllWindows()
