import numpy as np
import cv2 as cv

k3 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32); print("k3", k3)
img7 = np.arange(49).reshape((7, 7)); img7 = img7 * 1.0; print("img7\n", img7)
img9 = cv.copyMakeBorder(img7, 1, 1, 1, 1, cv.BORDER_REPLICATE); print("img9\n", img9)

img9k3 = cv.filter2D(img9, -1, k3); print("img9k3\n", img9k3)

k9 = cv.copyMakeBorder(k3, 3, 3, 3, 3, cv.BORDER_CONSTANT, value=0)
fk9_numpy = np.fft.fft2(k9); print("fk9_numpy\n", np.real(fk9_numpy))
fimg9_numpy = np.fft.fft2(img9); print("fimg9_numpy\n", np.real(fimg9_numpy))
fimg9fk9_numpy = np.multiply(fimg9_numpy, fk9_numpy); print("fimg9fk9_numpy\n", np.real(fimg9fk9_numpy))
ifimg9fk9_numpy = np.fft.ifft2(fimg9fk9_numpy); print("ifimg9fk9_numpy\n", np.real(ifimg9fk9_numpy))
sifimg9fk9_numpy = np.fft.fftshift(ifimg9fk9_numpy); print("sifimg9fk9_numpy", np.real(sifimg9fk9_numpy))

