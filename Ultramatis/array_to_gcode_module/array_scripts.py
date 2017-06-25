# Functions to create an array
import numpy as np


def qr_array():
    """Create a QR code array linking to ultramatis.com with error M (medium)"""
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

    # Add a line to the RHS (white then black) of 1px in diameter.
    x_size = np.shape(array)[0]
    array = np.append(array, np.zeros([x_size, 1]), axis=1)
    array = np.append(array, np.ones([x_size, 1]), axis=1)
    return array


def alphanumeric(n=8):
    """Generate n length alphanumeric QR code array"""
    import random
    from datetime import datetime
    random.seed(datetime.now())

    from string import digits, ascii_uppercase
    punctuation = ' $%*+-./:'
    id = ''.join(random.choice(digits + ascii_uppercase + punctuation) for _ in range(n))
    print(id)

    # Convert array to QR code
    import pyqrcode
    # Make qr code array
    url = pyqrcode.create(id, error='M')
    url.png('id.png')
    array = np.array(url.code)
    return array


def square():
    """Just a black pixel"""
    array = np.array([[1]])
    return array


def black_white():
    """Black then white pixel"""
    array = np.array([[1, 0]])
    return array
