import numpy as np
import pyqrcode

# CHANGE THIS TO THE PIXEL WIDTH IN mm
PIXEL_WIDTH = 0.5
# CHANGE THIS TO THE GRID WIDTH IN mm
DIVISOR = 6
# GRID_WIDTH = PIXEL_WIDTH / DIVISOR
GRID_WIDTH = 0.0865
# CHANGE THIS TO SPOT SIZE IN mm
SPOT_SIZE = 0.150


def array_to_gcode(array, save_name='qr'):
    def new_line():
        return 'G0 Z{0:.4f}\n'.format(GRID_WIDTH)

    def line(offset, length, speed=0.25, reverse=False):
        # Account for spot size
        dx = SPOT_SIZE/2

        # Flip directions if reversed
        if reverse:
            offset = -offset
            length = -length
            dx = -dx
        # Write G-Code
        res = ''
        if offset != 0:
            res += 'G0 X{0:.4f}\n'.format(offset * PIXEL_WIDTH)
        if length != 0:
            res += 'G0 X{0:.4f}\n'.format(dx)
            res += 'G0 Y1\n'
            res += 'G1 X{0:.4f} F{1}\n'.format(length * PIXEL_WIDTH - 2*dx, speed)
            res += 'G0 Y-1\n'
            res += 'G0 X{0:.4f}\n'.format(dx)
        return res

    def row_to_gcode(row, reverse=False):
        if reverse:
            row = row[::-1]
        offset = 0  # Set x-offset of the pen
        length = 0
        gcode = ''
        last_bit = 1
        for col_no, bit in enumerate(row):
            if bit != last_bit:
                if length:
                    gcode += line(offset, length, reverse=reverse)
                    offset = 0
                    length = 0
                last_bit = bit
            if bit == 1:
                length += 1
            else:
                offset += 1
        # Draw line for whatever is left in the row
        gcode += line(offset, length, reverse=reverse)
        return gcode

    coord = ''
    # Loop through each row of the code
    for row in array:
        for i in range(int(DIVISOR/2)):
            # Draw line forward
            coord += row_to_gcode(row)
            # Increment new line
            coord += new_line()
            # Draw line backward
            coord += row_to_gcode(row, reverse=True)
            # Increment new line
            coord += new_line()

    # Write to file
    with open(save_name + ".gcode", "w") as text_file:
        # Set to relative distance mode
        print('G91', file=text_file)
        # Set units to mm
        print('G21', file=text_file)
        # Set coordinate system to 0,0,0
        print('G92 X0 Y0 Z0\n', file=text_file)
        # Begin with laser focus 1mm off the sample so no ablation (drawing a line will refocus)
        print('G0 Y-1', file=text_file)
        # Write coordinates
        print(coord, file=text_file)
        # End program
        print('M2', file=text_file)


def qr_array():
    # Make qr code array
    url = pyqrcode.create('ultramatis.com', error='M')
    url.png('ultramatis.png')
    array = np.array(url.code)
    return array


def test_array():
    # Make Checkerboard
    array = np.array([[1, 0] * 4, [0, 1] * 4] * 2)
    array = np.append(array, np.zeros([1, 8]), axis=0)
    array = np.append(array, np.ones([1, 8]), axis=0)
    return array


if __name__ == "__main__":
    # Make Array
    # array = qr_array()
    array = test_array()

    # Plot Array
    # plt.imshow(array, cmap='Greys')
    # plt.show()

    # OPTIONAL: #  Flip array (invert)
    # array = 1 - array

    # size of the QR code in mm
    qr_size = [PIXEL_WIDTH * array.shape[0], PIXEL_WIDTH * array.shape[1]]
    print("QR Code x and y size: {1:.2f}, {0:.2f} mm".format(*qr_size))
    print("Pixels: {}".format(array.shape))

    # Write to G code
    array_to_gcode(array, save_name='test_array')

    # Online G-code viewer: http://nraynaud.github.io/webgcode/
