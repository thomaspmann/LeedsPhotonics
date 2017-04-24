import numpy as np
from numpy import deg2rad, sin, cos


class ArrayToGcode:
    def __init__(self, array, pixel_width=0.5, lines_per_pixel=6, speed=1.0, dy=1, spot_size=0):
        """
        :param array: Array to turn into gcode script
        :param pixel_width: pixel width in mm
        :param lines_per_pixel: how many lines per pixel
        :param speed: speed of writing (mm/s)
        :param dy: shift of the laser focus when not abalating target (-ve makes stage move away from focus)
        :param spot_size: size of the drawing spot in mm
        """
        self.array = np.array(array)
        self.pixel_width = pixel_width
        self.lines_per_pixel = lines_per_pixel
        self.speed = speed
        self.dy = dy
        self.spot_size = spot_size
        # Resolution of the grid
        self.grid_width = pixel_width / lines_per_pixel

    def invert(self):
        self.array = 1 - self.array

    def print_size(self):
        """print the size of the QR code in mm"""
        qr_size = [self.pixel_width * self.array.shape[0], self.pixel_width * self.array.shape[1]]
        print("QR Code x and y size: {1:.2f}, {0:.2f} mm".format(*qr_size))
        print("Pixels: {}".format(self.array.shape[::-1]))

    def plot_array(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.array, cmap='Greys')
        plt.show()

    # Functions for converting array to gcode
    def new_line(self):
        return 'G0 X{0:.4f}\n'.format(self.grid_width)

    def line(self, offset, length, reverse=False):
        # Account for spot size
        dz = self.spot_size / 2

        # Flip directions if reversed
        if reverse:
            offset = -offset
            length = -length
            dz = -dz

        # Write G-Code
        res = ''
        if offset != 0:
            res += 'G0 Z{0:.4f}\n'.format(offset * self.pixel_width)
        if length != 0:
            res += 'G0 Z{0:.4f}\n'.format(dz)
            res += 'G0 Y-{0:.2f}\n'.format(self.dy)
            res += 'G1 Z{0:.4f} F{1}\n'.format(length*self.pixel_width - 2*dz, self.speed)
            res += 'G0 Y{0:.2f}\n'.format(self.dy)
            res += 'G0 Z{0:.4f}\n'.format(dz)
        return res

    def row_to_gcode(self, row, reverse=False):
        if reverse:
            row = row[::-1]
        offset = 0  # Set x-offset of the pen
        length = 0
        gcode = ''
        last_bit = 1
        for col_no, bit in enumerate(row):
            if bit != last_bit:
                if length:
                    gcode += self.line(offset, length, reverse=reverse)
                    offset = 0
                    length = 0
                last_bit = bit
            if bit == 1:
                length += 1
            else:
                offset += 1
        # Draw line for whatever is left in the row
        gcode += self.line(offset, length, reverse=reverse)
        return gcode

    def to_file(self, save_name='qr'):
        # Create sting containing gcode instructions
        coord = ''
        # Loop through each row of the code
        for row in self.array:
            for i in range(int(self.lines_per_pixel/2)):
                # Draw line forward
                coord += self.row_to_gcode(row)
                # Increment new line
                coord += self.new_line()
                # Draw line backward
                coord += self.row_to_gcode(row, reverse=True)
                # Increment new line
                coord += self.new_line()
        self.save(coord, save_name)

    def save(self, coord, save_name):
        # Write to file
        with open(save_name + ".gcode", "w") as text_file:
            # Set to relative distance mode
            print('G91', file=text_file)
            # Set to metric (units in mm)
            print('G71', file=text_file)
            # Time units in seconds
            print('G76', file=text_file)
            # Set starting coordinate system to 0,0,0
            print('G92 X0 Y0 Z0\n', file=text_file)
            # Begin with laser focus off the target surface so no ablation (drawing a line will refocus)
            print('G0 Y{0:.2f}'.format(self.dy), file=text_file)
            # Write coordinates
            print(coord, file=text_file)
            # End with laser focus on the target surface again
            print('G0 Y-{0:.2f}\n'.format(self.dy), file=text_file)
            # End program
            print('M2', file=text_file)


class ArrayToGcodeAngled(ArrayToGcode):
    def __init__(self, degrees, array, pixel_width=0.5, lines_per_pixel=6, speed=1.0, spot_size=0):
        super().__init__(array, pixel_width, lines_per_pixel, speed, spot_size)
        # degrees is the angle between y axis and the target/substrate
        self.degrees = degrees

    def new_line(self):
        rad = deg2rad(self.degrees)
        dx = self.grid_width * sin(rad)
        dy = self.grid_width * cos(rad)
        return 'G0 X{0:.4f} Y{0:.4f}\n'.format(dx, dy)
