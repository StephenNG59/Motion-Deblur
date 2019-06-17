import cv2 as cv
import numpy as np
from image import *

img = cv.imread('img/nbt/man.png')
kernel = initial_kernel()

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

img_x, img_y, _ = img.shape
ker_x, ker_y = kernel.shape

b, g, r = cv.split(img)

ker_pad = cv.copyMakeBorder(kernel, 0, img_x - ker_x, 0, img_y - ker_y, cv.BORDER_CONSTANT)
cb = np.real(np.fft.ifft2(np.fft.fft2(b) / np.fft.fft2(ker_pad)))
cg = np.real(np.fft.ifft2(np.fft.fft2(g) / np.fft.fft2(ker_pad)))
cr = np.real(np.fft.ifft2(np.fft.fft2(r) / np.fft.fft2(ker_pad)))

print(img.shape)
print(ker_pad.shape)
c1 = cv.merge((cb, cg, cr))
# c1 = np.dstack([cb, cg, cr])
cv.imshow('c1', c1/255)
cv.waitKey(0)
cv.destroyAllWindows()

kernel = initial_kernel()
kernel = np.rot90(kernel, 2)
br = cv.filter2D(r, -1, kernel, borderType=cv.BORDER_REPLICATE)
bg = cv.filter2D(g, -1, kernel, borderType=cv.BORDER_REPLICATE)
bb = cv.filter2D(b, -1, kernel, borderType=cv.BORDER_REPLICATE)
# img_blur = cv.merge([r[1:y-1, 1:x-1], g[1:y-1, 1:x-1], b[1:y-1, 1:x-1]])
img_blur = cv.merge([bb, bg, br])
cv.imshow('blur', img_blur)
cv.waitKey(0)
cv.destroyAllWindows()
