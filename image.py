import numpy as np
import cv2 as cv


def load_image(file_path):
    im = cv.imread(file_path, cv.IMREAD_UNCHANGED)
    return im


def initial_kernel(size):
    assert size[0] == size[1], "error: currently only square kernels are supported"
    # kernel = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]])
    # kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
    # x = 11; kernel = np.ones((x, x), np.float32); kernel[x//2][x//2] = 1.0
    # kernel = np.arange(x * x).reshape((x, x))
    # kernel = np.ones((3, 3))
    # kernel = kernel / np.sum(kernel)
    # kernel = np.identity(size[0])
    kernel = cv.imread("./img/pcs/picassoBlurImage_kernel.png")
    kernel = kernel[:, :, 0]
    return kernel


def get_partial(image):
    """
    Computer the partial derivative in x and y direction.
    :param image: ndarray of shape [height, width], an image of uint8 type
    :return:
        par_x, par_y: ndarray of shape [height, width], partial in x, partial in y, in range (-255.0, 255.0)
    """
    im = image * 1.0                                # transform to float type, otherwise cannot use filter2D
    dx = np.array([[-1., 1.]], dtype=np.float32)    # should use float32 dtype, otherwise cannot use filter2D
    dy = np.array([[-1.], [1.]], dtype=np.float32)
    par_x = cv.filter2D(im, -1, dx, borderType=cv.BORDER_REPLICATE)                 # here border type is reflect (101?)
    par_y = cv.filter2D(im, -1, dy, borderType=cv.BORDER_REPLICATE)
    # region (not suitable code)
    # par_x = cv.Sobel(image, -1, dx=1, dy=0, ksize=3, borderType=cv.BORDER_DEFAULT)
    # par_y = cv.Sobel(image, -1, dx=0, dy=1, ksize=3, borderType=cv.BORDER_DEFAULT)
    # par_x = cv.convertScaleAbs(par_x)
    # par_y = cv.convertScaleAbs(par_y)
    # endregion
    return par_x, par_y


def get_smooth_mask(src, window_size, threshold):
    """
    Compute smooth region based on threshold.
    :param src: source image [width of image, height of image, channels]
    :param window_size: tuple object - [height of window, width of window]
    :param threshold: threshold of standard deviation
    :return:
        smooth_mask: a mask of the smooth region, same shape as src
    """
    img_h, img_w, img_c = src.shape
    win_h, win_w = window_size
    assert win_w % 2 == 1 & win_h % 2 == 1, "window size must be odd number"
    win_half_w, win_half_h = win_w // 2, win_h // 2

    # padding
    src_pad = cv.copyMakeBorder(src, win_half_h, win_half_h, win_half_w, win_half_w, cv.BORDER_REFLECT_101)

    sd = np.zeros(src.shape)        # [h, w, c]
    for i in range(img_h):
        if i == img_h // 2:
            print("  process: 50%...")
        y = i + win_half_h
        for j in range(img_w):
            x = j + win_half_w
            window = src_pad[i:y+win_half_h, j:x+win_half_w, :]
            for k in range(img_c):
                sd[i][j][k] = np.std(window[:, :, k])
    sd = np.sum(sd, axis=2)         # add up standard deviation of all channels

    smooth_mask = np.zeros((img_h, img_w, 1), np.uint8)
    for i in range(img_h):
        for j in range(img_w):
            # print(sd[i][j])
            if sd[i][j] < threshold:
                smooth_mask[i][j] = 0xff

    return smooth_mask


def pad(img, pad_y, pad_x, rot=0):
    i = np.rot90(img, rot)
    i = cv.copyMakeBorder(i, 0, pad_y, 0, pad_x, cv.BORDER_CONSTANT)
    return i


def fft(img):
    return np.fft.fft2(img)
