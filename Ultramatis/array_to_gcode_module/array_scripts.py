# Functions to create an array
import numpy as np


def qr_array():
    import pyqrcode
    # Make qr code array
    url = pyqrcode.create('ultramatis.com', error='M')
    url.png('ultramatis.png')
    array = np.array(url.code)
    return array


def test_array():
    """Make Checkerboard Array"""
    # Base array checker pattern
    array = np.array([[1, 0], [0, 1]])
    # Expand each array element to cover n pixels
    n = 4
    array = array.repeat(repeats=n, axis=0).repeat(repeats=n, axis=1)
    # Repeat the pattern to form a checkerboard
    array = np.tile(array, [1, 2])
    # Add two lines to the bottom (one white one black). Note the lines are 1 pixel in diameter.
    y_size = np.shape(array)[1]
    array = np.append(array, np.zeros([1, y_size]), axis=0)
    array = np.append(array, np.ones([1, y_size]), axis=0)
    array = np.append(array, np.zeros([1, y_size]), axis=0)
    array = np.append(array, np.ones([1, y_size]), axis=0)
    return array