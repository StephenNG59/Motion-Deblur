import numpy as np


def update_L(observed_image, kernel, psi):
    """

    :param observed_image:
    :param kernel:
    :param psi:
    :return:
    """

    d0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    dx = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
    dy = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    dxx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
    dyy = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32)
    dxy = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)



    # conjugate function
    np.conj()
    # fast fourier transform function
    np.fft.fft2()

    np.clip()
