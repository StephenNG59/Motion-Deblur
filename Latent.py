import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from vars import *


class Latent:

    d0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
    dx = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], np.float32)
    dy = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], np.float32)
    dxx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], np.float32)
    dyy = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], np.float32)
    dxy = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], np.float32)
    der_length = d0.shape[0]

    def __init__(self, observed_image, kernel_size, base=50):
        self.img = self.latent = observed_image

        self.img_y, self.img_x, _ = observed_image.shape
        self.ker_y, self.ker_x = kernel_size
        self.pad_y, self.pad_x = self.img_y + self.ker_y - 1, self.img_x + self.ker_x - 1

        d_pad_bottom = self.pad_y - self.der_length
        d_pad_right = self.pad_x - self.der_length
        self.d0_pad = cv.copyMakeBorder(self.d0, 0, d_pad_bottom, 0, d_pad_right, cv.BORDER_CONSTANT, value=0)
        self.dx_pad = cv.copyMakeBorder(self.dx, 0, d_pad_bottom, 0, d_pad_right, cv.BORDER_CONSTANT, value=0)
        self.dy_pad = cv.copyMakeBorder(self.dy, 0, d_pad_bottom, 0, d_pad_right, cv.BORDER_CONSTANT, value=0)
        self.dxx_pad = cv.copyMakeBorder(self.dxx, 0, d_pad_bottom, 0, d_pad_right, cv.BORDER_CONSTANT, value=0)
        self.dyy_pad = cv.copyMakeBorder(self.dyy, 0, d_pad_bottom, 0, d_pad_right, cv.BORDER_CONSTANT, value=0)
        self.dxy_pad = cv.copyMakeBorder(self.dxy, 0, d_pad_bottom, 0, d_pad_right, cv.BORDER_CONSTANT, value=0)

        self.f_d0_pad = np.fft.fft2(self.d0_pad)
        self.f_dx_pad = np.fft.fft2(self.dx_pad)
        self.f_dy_pad = np.fft.fft2(self.dy_pad)
        self.f_dxx_pad = np.fft.fft2(self.dxx_pad)
        self.f_dyy_pad = np.fft.fft2(self.dyy_pad)
        self.f_dxy_pad = np.fft.fft2(self.dxy_pad)

        self.Delta = base * (1 * (np.conjugate(self.f_d0_pad) * self.f_d0_pad) +
                             0.5 * (np.conjugate(self.f_dx_pad) * self.f_dx_pad +
                                    np.conjugate(self.f_dy_pad) * self.f_dy_pad) +
                             0.25 * (np.conjugate(self.f_dxx_pad) * self.f_dxx_pad +
                                     np.conjugate(self.f_dyy_pad) * self.f_dyy_pad +
                                     np.conjugate(self.f_dxy_pad) * self.f_dxy_pad))

    def update_l(self, kernel, psi):
        """

        :param kernel:
        :param psi:
        :return:
        """

        observed_r, observed_g, observed_b = cv.split(self.img)

        l_updated_r = self.update_l_channel(observed_r, kernel, (psi[0][:, :, 0], psi[1][:, :, 0]))
        l_updated_g = self.update_l_channel(observed_g, kernel, (psi[0][:, :, 1], psi[1][:, :, 1]))
        l_updated_b = self.update_l_channel(observed_b, kernel, (psi[0][:, :, 2], psi[1][:, :, 2]))

        l_updated = np.dstack([l_updated_r, l_updated_g, l_updated_b]) / 255.0
        plt.subplot(111); plt.imshow(l_updated); plt.show()

        l_diff = l_updated - self.latent
        diff_r = np.linalg.norm(l_diff[:, :, 0], ord=2)
        diff_g = np.linalg.norm(l_diff[:, :, 1], ord=2)
        diff_b = np.linalg.norm(l_diff[:, :, 2], ord=2)
        diff = max(diff_r, diff_g, diff_b)

        self.latent = l_updated

        return diff

    def update_l_channel(self, observed_image_channel, kernel, psi):
        """

        :param observed_image_channel:
        :param kernel:
        :param psi:
        :return:
        """
        if kernel.shape != (self.ker_x, self.ker_y):
            self.ker_x, self.ker_y = kernel.shape
            self.pad_x, self.pad_y = self.img_x + self.ker_x - 1, self.img_y + self.ker_y - 1

        img = observed_image_channel

        # fft of padded psi
        psi_x, psi_y = psi
        psi_x_pad = cv.copyMakeBorder(psi_x, 0, self.ker_y - 1, 0, self.ker_x - 1, cv.BORDER_CONSTANT, value=0)
        psi_y_pad = cv.copyMakeBorder(psi_y, 0, self.ker_y - 1, 0, self.ker_x - 1, cv.BORDER_CONSTANT, value=0)
        f_psi_x_pad, f_psi_y_pad = np.fft.fft2(psi_x_pad), np.fft.fft2(psi_y_pad)

        # fft of padded kernel
        kernel_pad = cv.copyMakeBorder(kernel, 0, self.img_y - 1, 0, self.img_x - 1, cv.BORDER_CONSTANT, value=0)
        f_kernel_pad = np.fft.fft2(kernel_pad)

        # fft of padded img
        img_pad = cv.copyMakeBorder(img, 0, self.ker_y - 1, 0, self.ker_x - 1, cv.BORDER_CONSTANT, value=0)
        f_img_pad = np.fft.fft2(img_pad)

        f_dx_pad = np.fft.fft2(self.dx_pad)
        f_dy_pad = np.fft.fft2(self.dy_pad)

        plt.subplot(121); plt.imshow(img, cmap='gray');
        l_channel_updated = np.fft.ifft2(
            (np.conjugate(f_kernel_pad) * f_img_pad * self.Delta
             + Gamma * np.conjugate(f_dx_pad) * f_psi_x_pad
             + Gamma * np.conjugate(f_dy_pad) * f_psi_y_pad
             ) /
            (
             np.conjugate(f_kernel_pad) * f_kernel_pad * self.Delta
             + Gamma * np.conjugate(f_dx_pad) * f_dx_pad
             + Gamma * np.conjugate(f_dy_pad) * f_dy_pad
             ))
        plt.subplot(122); plt.imshow(np.real(l_channel_updated[:self.img_y, :self.img_x]), cmap='gray');
        plt.show()

        return np.real(l_channel_updated[:self.img_y, :self.img_x])
