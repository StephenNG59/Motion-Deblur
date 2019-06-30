import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from vars import *
from image import pad, fft
from scipy.fftpack import ifft2, ifftshift, fftshift


class Latent:

    d0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
    dx = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], np.float32)
    dy = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], np.float32)
    dxx = np.zeros((5, 5))
    dxx[2][2] = -0.5
    dxx[2][0] = 0.25
    dxx[2][4] = 0.25
    dyy = np.zeros((5, 5))
    dxx[2][2] = -0.5
    dxx[0][2] = 0.25
    dxx[4][2] = 0.25
    dxy = np.array([[-0.25, 0, 0.25], [0, 0, 0], [0.25, 0, -0.25]], np.float32)
    der_length1 = d0.shape[0]
    der_length2 = dxx.shape[0]

    def __init__(self, observed_image, kernel_size, base=50):
        self.img = self.latent = observed_image

        self.img_y, self.img_x, _ = observed_image.shape
        self.ker_y, self.ker_x = kernel_size
        self.pad_y, self.pad_x = self.img_y, self.img_x

        d_pad_bottom1 = self.pad_y - self.der_length1
        d_pad_right1 = self.pad_x - self.der_length1
        d_pad_bottom2 = self.pad_y - self.der_length2
        d_pad_right2 = self.pad_x - self.der_length2

        self.f_d0_pad = fft(pad(self.d0, d_pad_bottom1, d_pad_right1, 2))
        self.f_dx_pad = fft(pad(self.dx, d_pad_bottom1, d_pad_right1, 2))
        self.f_dy_pad = fft(pad(self.dy, d_pad_bottom1, d_pad_right1, 2))
        self.f_dxx_pad = fft(pad(self.dxx, d_pad_bottom2, d_pad_right2, 2))
        self.f_dyy_pad = fft(pad(self.dyy, d_pad_bottom2, d_pad_right2, 2))
        self.f_dxy_pad = fft(pad(self.dxy, d_pad_bottom1, d_pad_right1, 2))

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

        observed_b, observed_g, observed_r = cv.split(self.img)

        l_updated_b = self.update_l_channel(observed_b, kernel, (psi[0][:, :, 2], psi[1][:, :, 2])) / 255.0
        l_updated_g = self.update_l_channel(observed_g, kernel, (psi[0][:, :, 1], psi[1][:, :, 1])) / 255.0
        l_updated_r = self.update_l_channel(observed_r, kernel, (psi[0][:, :, 0], psi[1][:, :, 0])) / 255.0

        l_updated = np.dstack([l_updated_b, l_updated_g, l_updated_r])

        plt.subplot(111); plt.title("Latent updated")
        plt.imshow(np.dstack([l_updated_r, l_updated_g, l_updated_b])); plt.show()

        l_diff = l_updated - self.latent
        diff_b = np.linalg.norm(l_diff[:, :, 0], ord=1)
        diff_g = np.linalg.norm(l_diff[:, :, 1], ord=1)
        diff_r = np.linalg.norm(l_diff[:, :, 2], ord=1)
        diff = max(diff_r, diff_g, diff_b)

        self.latent = l_updated
        # l_updated[]
        cv.imwrite('img/latent/l.jpg', 255 * l_updated)
        cv.imshow('l_updated', l_updated)
        cv.waitKey(0)
        cv.destroyAllWindows()


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
        f_psi_x_pad = fft(pad(psi_x, 0, 0))
        f_psi_y_pad = fft(pad(psi_y, 0, 0))

        # fft of padded kernel
        f_kernel_pad = fft(pad(kernel, self.img_y - self.ker_y, self.img_x - self.ker_x))

        # fft of padded img
        f_img_pad = fft(img)

        # plt.subplot(121); plt.imshow(img, cmap='gray');

        shift_kernel = np.zeros((self.img_y, self.img_x))
        shift_kernel[self.ker_y//2][self.ker_x//2] = 1
        print(shift_kernel.shape)

        # print(Gamma * np.conjugate(self.f_dx_pad) * f_psi_x_pad)
        # print("-----------------    ")
        # print(Gamma * np.conjugate(self.f_dx_pad) * self.f_dx_pad)

        l_channel_updated = ifft2(
            fft(shift_kernel) *
            (np.conjugate(f_kernel_pad) * f_img_pad * self.Delta
             + Gamma * np.conjugate(self.f_dx_pad) * f_psi_x_pad
             + Gamma * np.conjugate(self.f_dy_pad) * f_psi_y_pad
             ) /
            (
             np.conjugate(f_kernel_pad) * f_kernel_pad * self.Delta
             + Gamma * np.conjugate(self.f_dx_pad) * self.f_dx_pad
             + Gamma * np.conjugate(self.f_dy_pad) * self.f_dy_pad
             ))

        # plt.subplot(122); plt.imshow(np.real(l_channel_updated[:self.img_y, :self.img_x]), cmap='gray');
        plt.show()

        # print(observed_image_channel.shape)
        # print(l_channel_updated.shape)
        # print(self.ker_x//2)
        # return np.real(l_channel_updated[self.ker_y//2:self.img_y+self.ker_y//2,
        # self.ker_x//2:self.img_x+self.ker_x//2])
        # return np.real(l_channel_updated[:self.img_y, :self.img_x])
        return np.real(l_channel_updated)


if __name__ == '__main__':
    pass
