import numpy as np
from image import get_partial
from qpsolvers import solve_qp
import conv_to_mul as ctm
import cv2 as cv
# from memory_profiler import profile


def update_f(latent1, image1, f):

    A0 = 50**0.5 * ctm.conv_to_mul(latent1, f)
    B0 = 50 ** 0.5 * image1.reshape((1, -1))

    dx = np.array([[-1., 1.]])
    latent_x = cv.filter2D(latent1, -1, dx, borderType=cv.BORDER_REPLICATE)
    Ax = 25 ** 0.5 * ctm.conv_to_mul(latent_x, f)
    image_x = cv.filter2D(image1, -1, dx, borderType=cv.BORDER_REPLICATE)
    Bx = 25 ** 0.5 * image_x.reshape((1, -1))

    dy = np.array([[-1.], [1.]])
    latent_y = cv.filter2D(latent1, -1, dy, borderType=cv.BORDER_REPLICATE)
    Ay = 25 ** 0.5 * ctm.conv_to_mul(latent_y, f)
    image_y = cv.filter2D(image1, -1, dy, borderType=cv.BORDER_REPLICATE)
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

    P = 2 * np.dot(A.T, A)
    q = 1-2 * np.dot(B.T, A)
    q = q.flatten()

    # G1 = np.identity(q.size)
    # h1 = np.ones(q.size)
    G2 = -1 * np.identity(q.size)
    h2 = np.zeros(q.size)
    # G = np.concatenate((G1, G2))
    # h = np.concatenate((h1, h2))
    G = G2
    h = h2

    updated_f = solve_qp(P, q, G, h, np.zeros((q.size, q.size)), np.zeros(q.size))
    updated_f = updated_f.reshape(f.shape)
    max_f = updated_f.max()
    rows, cols = updated_f.shape
    for i in range(rows):
        for j in range(cols):
            if updated_f[i][j] < max_f * 0.3:
                updated_f[i][j] = 0
    cv.imwrite('img/kernel/kernel.png', updated_f * 255)
    updated_f = updated_f / np.sum(updated_f)

    cv.imwrite('img/kernel/kernel_norm.png', updated_f * 255)

    return updated_f


if __name__ == '__main__':
    mat1 = np.eye(4, 3)
    mat2 = np.ones((4, 3))

    latent = cv.merge((mat1, mat1, mat1))
    image = cv.merge((mat1, mat2, mat1))

    f = np.ones((3, 3))
    f = f / sum(f)
    print(update_f(latent, image, f))
