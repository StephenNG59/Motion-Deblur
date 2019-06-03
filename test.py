import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

file_path = "./img/asc.jpg"
window_name = "ASC"

# im = cv.imread("./img/asc.jpg", cv.IMREAD_GRAYSCALE)
im = cv.imread(file_path, cv.IMREAD_UNCHANGED)        # cv.IMREAD_COLOR
# cv.namedWindow(window_name, cv.WINDOW_NORMAL)

b, g, r = cv.split(im)
im2 = cv.merge([r, g, b])
# plt.subplot(121); plt.imshow(im)    # expects distorted color
plt.subplot(211); plt.imshow(im2)   # expect true color

d_depth = cv.CV_16S

dx = cv.Sobel(im2, ddepth=d_depth, dx=1, dy=0, ksize=3, scale=15, borderType=cv.BORDER_DEFAULT)
dy = cv.Sobel(im2, ddepth=d_depth, dx=0, dy=1, ksize=3, borderType=cv.BORDER_DEFAULT)
# dxy = np.zeros(dx.shape)

dx = cv.convertScaleAbs(dx)
dy = cv.convertScaleAbs(dy)
dxy = cv.addWeighted(dx, 0.5, dy, 0.5, 0)
dxy = cv.convertScaleAbs(dxy)

plt.subplot(231); plt.imshow(dx)
plt.subplot(232); plt.imshow(dy)
plt.subplot(233); plt.imshow(dxy)

plt.show()

# while True:
#     cv.imshow(window_name, im)
#     key = cv.waitKey(0) & 0xFF
#
#     if key == 27:        # ESC
#         break
#     elif key == ord("s"):
#         cv.imwrite(file_path + "_gray.jpg", im)
#         break
#     elif key == ord('a'):
#         # 反色
#         im = 255 - im
#         cv.imwrite(file_path + "_inverse.jpg", im)
#         break
#     elif key == ord('b'):
#         # 分色
#         b = np.zeros(im.shape)
#         g = np.zeros(im.shape)
#         r = np.zeros(im.shape)
#         b[:, :, 0], g[:, :, 1], r[:, :, 2] = cv.split(im)
#         cv.imwrite(file_path + "_b.jpg", b)
#         cv.imwrite(file_path + "_g.jpg", g)
#         cv.imwrite(file_path + "_r.jpg", r)
#         break


# cv.destroyAllWindows()

