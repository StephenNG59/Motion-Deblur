import cv2 as cv


def load_image(file_path):
    im = cv.imread(file_path, cv.IMREAD_UNCHANGED)
    return im


def get_partial(image):
    d_depth = cv.CV_16S
    par_x = cv.Sobel(image, ddepth=d_depth, dx=1, dy=0, ksize=3, borderType=cv.BORDER_DEFAULT)
    par_y = cv.Sobel(image, ddepth=d_depth, dx=0, dy=1, ksize=3, borderType=cv.BORDER_DEFAULT)
    par_x = cv.convertScaleAbs(par_x)
    par_y = cv.convertScaleAbs(par_y)
    return par_x, par_y
