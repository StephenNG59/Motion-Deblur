import cv2 as cv
import numpy as np


dy = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], np.float32)
dx = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])

img = cv.imread('img/out/man.png')
cv.imshow('man', img)
cv.waitKey(0)
cv.destroyAllWindows()
print(img.shape)
img = cv.filter2D(img, -1, dx, borderType=cv.BORDER_REPLICATE)
img *= 30
# print(img)
cv.imwrite('img/dx/dx_man.png', img)

latent = cv.imread('img/out/out-1.jpg')
cv.imshow('out-1', latent)
cv.waitKey(0)
cv.destroyAllWindows()
print(latent.shape)
latent = cv.filter2D(latent, -1, dx, borderType=cv.BORDER_REPLICATE)
latent *= 30
# print(latent)
cv.imwrite('img/dx/dx_latent.png', latent)

clear = cv.imread('img/out/man_clear.png')
cv.imshow('man_clear', clear)
cv.waitKey(0)
cv.destroyAllWindows()
print(clear.shape)
clear = cv.filter2D(clear, -1, dx, borderType=cv.BORDER_REPLICATE)
clear *= 30
# print(latent)
cv.imwrite('img/dx/dx_man_clear.png', clear)

img = cv.filter2D(img, -1, dy, borderType=cv.BORDER_REPLICATE)
img *= 30
# print(img)
cv.imwrite('img/dx/dy_man.png', img)

latent = cv.filter2D(latent, -1, dy, borderType=cv.BORDER_REPLICATE)
latent *= 30
# print(latent)
cv.imwrite('img/dx/dy_latent.png', latent)

clear = cv.filter2D(clear, -1, dy, borderType=cv.BORDER_REPLICATE)
clear *= 30
# print(latent)
cv.imwrite('img/dx/dy_man_clear.png', clear)
