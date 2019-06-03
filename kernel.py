import numpy as np


def initial_kernel(size):

    # currently only supports square
    assert size[0] == size[1], "error: kernel size is not square"

    kernel = np.identity(size[0])
    # kernel = np.eye(size[0])

    return kernel
