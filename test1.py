import numpy as np
import cv2 as cv

k3 = np.array([[0, 0, 0], [-0.5, 1, -0.5], [0, 0, 0]], dtype=np.float32); print("k3", k3)
# k3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32); print("k3", k3)

img3 = np.array([[0.1, 0.5, 0.2], [0.8, 0.3, 0.1], [0.2, 0.4, 0.3]], np.float32); print("img3", img3)
img5 = cv.copyMakeBorder(img3, 1, 1, 1, 1, cv.BORDER_REPLICATE); print("img5", img5)

img5k3 = cv.filter2D(img5, -1, k3); print("img5k3", img5k3)

print("\n-- 1. 对kernel先pad后fft --")
k5 = cv.copyMakeBorder(k3, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
# -- OPENCV
# fk5 = cv.dft(k5); print("fk5", fk5)
# fimg5 = cv.dft(img5); print("fimg5", fimg5)
# fimg5fk5 = cv.mulSpectrums(fimg5, fk5, flags=0); print("fimg5fk5", fimg5fk5)
# ifimg5fk5 = cv.dft(fimg5fk5, flags=cv.DFT_INVERSE + cv.DFT_SCALE); print("ifimg5fk5", ifimg5fk5)
# -- NUMPY
fk5_numpy = np.fft.fft2(k5); print("fk5_numpy", fk5_numpy)
fimg5_numpy = np.fft.fft2(img5); print("fimg5_numpy", fimg5_numpy)
fimg5fk5_numpy = np.multiply(fimg5_numpy, fk5_numpy); print("fimg5fk5_numpy", fimg5fk5_numpy)
ifimg5fk5_numpy = np.fft.ifft2(fimg5fk5_numpy); print("ifimg5fk5_numpy", ifimg5fk5_numpy)
sifimg5fk5_numpy = np.fft.fftshift(ifimg5fk5_numpy); print("shifted_numpy", sifimg5fk5_numpy)

print("\n-- 2. 对kernel先fft后pad --")
fk3 = cv.dft(k3)
fk5 = cv.copyMakeBorder(fk3, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0); print("fk5", fk5)
fimg5 = cv.dft(img5); print("fimg5", fimg5)
fimg5fk5 = cv.mulSpectrums(fimg5, fk5, flags=0); print("fimg5fk5", fimg5fk5)
ifimg5fk5 = cv.dft(fimg5fk5, flags=cv.DFT_INVERSE + cv.DFT_SCALE); print("ifimg5fk5", ifimg5fk5)



# f_k = cv.dft(k3)
# f_img = cv.dft(img3)
# fimg_fk = cv.mulSpectrums(f_img, f_k, 0)
# print("fimg_fk", fimg_fk)
#
# in_fimg_fk = cv.dft(fimg_fk, flags=cv.DFT_INVERSE + cv.DFT_SCALE)
# print("in_fimg_fk", in_fimg_fk)

