import numpy as np

k = 2.7
a = 6.1e-4
b = 5.0
l_t = (k + np.sqrt(k * k - 4 * a * b)) / (2 * a)       # -kl = -(a * l^2 + b) --> al^2 - kl + b = 0


def big_phi(x):
    """
    Logarithmic gradient density function
    :param x: gradient
    :return: the logarithmic density of the gradient
    """
    if abs(x) <= l_t:
        return -k * abs(x)
    else:
        return -(a * x * x + b)


def get_cof():
    return k, a, b, l_t
