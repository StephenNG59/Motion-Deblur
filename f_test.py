import numpy as np
from image import get_partial
from qpsolvers import solve_qp
import conv_to_mul as ctm
import cv2 as cv


mat1 = np.eye(4, 3)
mat2 = np.ones((4, 3))

latent = cv.merge((mat1, mat1, mat1))
image = cv.merge((mat1, mat2, mat1))

f = np.ones((3, 3))
f = f / sum(f)

print(ctm.conv_to_mul(latent, f))

A0 = 50 ** 0.5 * ctm.conv_to_mul(latent, f)
B0 = 50 ** 0.5 * image.reshape((1, -1))

print(A0.shape)
print(B0)
print(B0.shape)

dx = np.array([[-1., 1.]], dtype=np.float32)
latent_x = cv.filter2D(latent, -1, dx, borderType=cv.BORDER_REPLICATE)
Ax = 25 ** 0.5 * ctm.conv_to_mul(latent_x, f)
image_x = cv.filter2D(image, -1, dx, borderType=cv.BORDER_REPLICATE)
Bx = 25 ** 0.5 * image_x.reshape((1, -1))

dy = np.array([[-1.], [1.]], dtype=np.float32)
latent_y = cv.filter2D(latent, -1, dy, borderType=cv.BORDER_REPLICATE)
Ay = 25 ** 0.5 * ctm.conv_to_mul(latent_y, f)
image_y = cv.filter2D(image, -1, dy, borderType=cv.BORDER_REPLICATE)
By = 25 ** 0.5 * image_y.reshape((1, -1))

latent_xx = cv.filter2D(latent_x, -1, dx, borderType=cv.BORDER_REPLICATE)
Axx = 12.5 ** 0.5 * ctm.conv_to_mul(latent_xx, f)
image_xx = cv.filter2D(image_x, -1, dx, borderType=cv.BORDER_REPLICATE)
Bxx = 12.5 ** 0.5 * image_xx.reshape((1, -1))

latent_xy = cv.filter2D(latent_y, -1, dx, borderType=cv.BORDER_REPLICATE)
Axy = 12.5 ** 0.5 * ctm.conv_to_mul(latent_xy, f)
image_xy = cv.filter2D(image_y, -1, dx, borderType=cv.BORDER_REPLICATE)
Bxy = 12.5 ** 0.5 * image_xy.reshape((1, -1))

latent_yy = cv.filter2D(latent_y, -1, dy, borderType=cv.BORDER_REPLICATE)
Ayy = 12.5 ** 0.5 * ctm.conv_to_mul(latent_yy, f)
image_yy = cv.filter2D(image_y, -1, dy, borderType=cv.BORDER_REPLICATE)
Byy = 12.5 ** 0.5 * image_yy.reshape((1, -1))

A = np.concatenate((A0, Ax, Ay, Axx, Axy, Ayy))
B = np.concatenate((B0.T, Bx.T, By.T, Bxx.T, Bxy.T, Byy.T))

print(A.shape)
print(B.shape)

P = 2 * np.dot(A.T, A)
q = 1-2 * np.dot(B.T, A)
q = q.flatten()

print(q)
print(P.shape)

G1 = np.identity(q.size)
h1 = np.ones(q.size)
G2 = -1 * np.identity(q.size)
h2 = np.zeros(q.size)
G = np.concatenate((G1, G2))
h = np.concatenate((h1, h2))
print(G)
print(h)
updated_f = solve_qp(P, q, G, h, np.zeros((q.size, q.size)), np.zeros(q.size))
updated_f = updated_f.reshape(f.shape)
print("QP solution:", updated_f)
