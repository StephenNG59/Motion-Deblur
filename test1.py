import numpy as np
import cv2 as cv

k = np.array([[0, 0, 0, 0, 0], [-1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)

f_k = cv.dft(k)

img = np.array([[0.01, 0.06, 0.24, 0, 0], [0.80, 0.23, 0.01, 0, 0], [0.12, 0.43, 0.32, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32)
f_img = cv.dft(img)

img_k = cv.filter2D(img, -1, k)
print("img_k", img_k)

fimg_fk = cv.mulSpectrums(f_img, f_k, 0)
print("fimg_fk", fimg_fk)

in_fimg_fk = cv.dft(fimg_fk, flags=cv.DFT_INVERSE + cv.DFT_SCALE)
print("in_fimg_fk", in_fimg_fk)

