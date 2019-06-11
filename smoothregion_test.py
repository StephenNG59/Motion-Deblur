import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
# from image import *
from image import *
import time
from vars import *
from find_smooth_region import *

# img1 = load_image('./img/man.png')
img1 = cv.imread('img/man.png')
print(img1.shape)
kernel = initial_kernel()
print(kernel.shape)

t1 = time.time()
print("1. Getting smooth region by method 1...")
smooth_mask = get_smooth_mask(img1, kernel.shape, smooth_threshold)
print("- 1.finished time {0}".format(time.time() - t1))
cv.imwrite("./img/out/sr_1.jpg", smooth_mask)

t2 = time.time()
print("2. Getting smooth region by method 2...")
smooth_mask = find_smooth_region(img1, kernel.shape, smooth_threshold)
print("- 2.finished time {0}".format(time.time() - t2))
cv.imwrite("./img/out/sr_2.jpg", smooth_mask)
