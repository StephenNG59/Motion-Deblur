import numpy as np
import cv2 as cv


file_path = "./img/test.jpg"
out_path = "./img/test_blur.jpg"

# kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, 4, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]], np.float32)

# k_w = 15
# k_h = 15
# kernel = np.ones((k_h, k_w))
# # kernel = np.zeros((k_h, k_w))
# # for i in range(k_w//2-10, k_w//2+11):
# #     kernel[k_h//2][i] = 0.05
# # kernel[k_h//2][k_w//2] = 0.1
x = 11; kernel = np.ones((x, x), np.float32); kernel[x//2][x//2] = 1.0
kernel = np.arange(x * x).reshape((x, x))
pad = kernel.shape[0] // 2
kernel = kernel / np.sum(kernel)


img = cv.imread(file_path)
y, x, _ = img.shape
# img_pad = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
r, g, b = cv.split(img)

r = cv.filter2D(r, -1, kernel, borderType=cv.BORDER_REPLICATE)
g = cv.filter2D(g, -1, kernel, borderType=cv.BORDER_REPLICATE)
b = cv.filter2D(b, -1, kernel, borderType=cv.BORDER_REPLICATE)
# img_blur = cv.merge([r[1:y-1, 1:x-1], g[1:y-1, 1:x-1], b[1:y-1, 1:x-1]])
img_blur = cv.merge([r, g, b])
cv.imwrite(out_path, img_blur)
