import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image import get_smooth_mask


file_path = "./img/test.png"
window_name = "ASC"

im = cv.imread(file_path, cv.IMREAD_UNCHANGED)        # cv.IMREAD_COLOR
mask = get_smooth_mask(im, (11, 11), threshold=5)
# cv.imshow("mask", mask)
# cv.waitKey(0)
# print(mask.shape)
# im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
# print(im.shape)
smooth = cv.bitwise_and(im, im, mask=mask)

plt.subplot(231); plt.imshow(im)
plt.subplot(232); plt.imshow(mask)
plt.subplot(233); plt.imshow(smooth)
plt.show()
