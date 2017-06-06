import numpy as np
from numpy import deg2rad, sin, cos, pi


class ArrayToGcode:
    def __init__(self,
                 array,
                 pixel_width=0.5,
                 lines_per_pixel=6,
                 speed=1.0,
                 d_focus=1.0,
                 spot_size=0,
                 focal_offset=0.8,
                 final_offset=5.0,
                 degrees=0.0):
        """
        :param array: Array to turn into gcode script
        :param pixel_width: pixel width in mm
        :param lines_per_pixel: how many lines per pixel
        :param speed: speed of writing (mm/s)
        :param d_focus: shift of the laser focus when not abalating target (-ve makes stage move away from focus)
        :param spot_size: size of the drawing spot in mm (subtract half of this off each side when drawing lines)
        :param focal_offset: offset of the laser focus from the target when writing
        :param final_offset: Final offset to move the laser focus when finished writing
        :param degrees: angle between the laser and the target surface
        """
        self.array = np.array(array)
        self.pixel_width = pixel_width
        self.lines_per_pixel = lines_per_pixel
        self.speed = speed
        self.spot_size = spot_size
        self.d_focus = d_focus
        self.focal_offset = focal_offset
        self.final_offset = final_offset

        # Resolution of the grid (when flat)
        self.grid_width = pixel_width / lines_per_pixel
        # Angle of implantation (between laser and target surface)
        self.rad = deg2rad(degrees)

    def invert(self):
        self.array = 1 - self.array

    def print_size(self):
        """print the size of the QR code in mm"""
        qr_size = [self.pixel_width * self.array.shape[0], self.pixel_width * self.array.shape[1]]
        print("QR Code x and y size: {1:.2f}, {0:.2f} mm".format(*qr_size))
        print("Pixels: {}".format(self.array.shape[::-1]))

    def plot_array(self, save_name='../array', show=True):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.array, cmap='Greys')
        plt.savefig(save_name)
        if show:
            plt.show()

    # Functions for converting array to gcode
    def new_line(self):
        if self.rad:
            dx = self.grid_width * sin(pi / 2 - self.rad)
            dy = self.grid_width * cos(pi / 2 - self.rad)
            return 'G0 X{0:.4f} Y{1:.4f}\n'.format(dx, dy)
        else:
            return 'G0 X{0:.4f}\n'.format(self.grid_width)

    def line(self, offset, length, reverse=False):
        # Shift focus when going from white to black - perpendicular to target hence sin and cos switched
        dx = self.d_focus * sin(-self.rad)
        dy = self.d_focus * cos(-self.rad)

        # Padding of black lines to account for spot size
        pad = self.spot_size / 2

        # Flip directions if reversed
        if reverse:
            offset = -offset
            length = -length
            pad = -pad

        # Write G-Code
        res = ''
        if offset:
            res += 'G1 Z{0:.4f} F10\n'.format(offset * self.pixel_width)
        if length:
            if pad:
                res += 'G1 Z{0:.4f} F10\n'.format(pad)
            # Bring laser into focus for drawing
            res += 'G1 X{0:.4f} Y{1:.4f} F10\n'.format(-dx, -dy)
            # Draw line
            res += 'G1 Z{0:.4f} F{1:.4f}\n'.format(length * self.pixel_width - 2 * pad, self.speed)
            # Move laser out of focus for no drawing
            res += 'G1 X{0:.4f} Y{1:.4f} F10\n'.format(dx, dy)
            if pad:
                res += 'G1 Z{0:.4f} F10\n'.format(pad)
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
        # Loop through each row of the code to create the coordinates
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

            # Write to file
            with open(save_name + ".gcode", "w") as text_file:
                # Set to relative distance mode
                print('G91', file=text_file)
                # Set to metric (units in mm)
                print('G71', file=text_file)
                # Time units in seconds
                print('G76', file=text_file)
                # Set starting coordinate system to (0,0,0)
                print('G92 X0 Y0 Z0\n', file=text_file)
                # Move the laser focus off the surface of the bottle to distance required for drawing and not etching
                if self.focal_offset:
                    dx = self.focal_offset * sin(self.rad)
                    dy = self.focal_offset * cos(self.rad)
                    print('G1 X{0:.4f} Y{1:.4f} F10'.format(-dx, -dy), file=text_file)
                # Begin with laser focus off the target surface so no drawing/ablation (drawing a line will refocus)
                # Note perpendicular to target hence sin and cos switched
                if self.d_focus:
                    dx = self.d_focus * sin(-self.rad)
                    dy = self.d_focus * cos(-self.rad)
                    print('G1 X{0:.4f} Y{1:.4f} F10'.format(dx, dy), file=text_file)
                # Write coordinates
                print(coord, file=text_file)
                # End with laser focus on the target surface again
                if self.d_focus:
                    dx = self.d_focus * sin(self.rad)
                    dy = self.d_focus * cos(self.rad)
                    print('G1 X{0:.4f} Y{1:.4f} F10\n'.format(-dx, -dy), file=text_file)
                # Move the laser away from the bottle
                if self.final_offset:
                    dx = self.final_offset * sin(self.rad)
                    dy = self.final_offset * cos(self.rad)
                    print('G1 X{0:.4} Y{1:.4} F7'.format(-dx, -dy), file=text_file)
                # End program
                print('M2', file=text_file)


class ArrayOfLines:
    def __init__(self, nlines=1, speed=1.0, dx=1.0, dy=1.0, dz=1.0, focal_offset=0.8, final_offset=0.0):
        """
        :param nlines: Number of lines to draw
        :param speed: speed at which to draw the lines
        :param dx: spacing of the lines
        :param dy: shift of laser focus when we don't want to draw 
        :param dz: length of the line
        :param focal_offset: position of the focus above the substrate when drawing
        :param final_offset: final offset of the focus when program is finished
        """
        self.nlines = nlines
        self.speed = speed
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.focal_offset = focal_offset
        self.final_offset = final_offset

    # Functions for converting array to gcode
    def new_line(self):
        # Shift dx across
        res = 'G0 X{0:.4f}\n'.format(self.dx)
        # Return back a blank line
        res += 'G0 Z{0:.4f}\n'.format(-self.dz)
        return res

    def draw_line(self):
        # Go to focus for drawing
        res = 'G0 Y{0:.4f}\n'.format(self.dy)
        # Draw line
        res += 'G1 Z{0:.4f} F{1:.4f}\n'.format(self.dz, self.speed)
        # Go out of focus again
        res += 'G0 Y{0:.4f}\n'.format(-self.dy)
        return res

    def array(self):
        # Create sting containing gcode instructions
        gcode = ''
        # Loop through each row of the code
        for i in range(self.nlines):
            # Draw black line
            gcode += self.draw_line()
            # Reset for new line
            gcode += self.new_line()
        return gcode

    def sweep_speed(self, speed_inc):
        # Create sting containing gcode instructions
        gcode = ''
        # Loop through each row of the code
        for _ in range(self.nlines):
            # Draw black line
            gcode += self.draw_line()
            # Reset for new line
            gcode += self.new_line()
            # Increment speed for next line
            self.speed += speed_inc
        return gcode

    def sweep_height(self, dy_inc):
        # Create sting containing gcode instructions
        gcode = ''
        # Loop through each row of the code
        for _ in range(self.nlines):
            # Draw black line
            gcode += self.draw_line()
            # Reset for new line
            gcode += self.new_line()
            # Increment height for next line
            gcode += 'G0 Y{0:.4f}\n'.format(-dy_inc)
        return gcode

    def save(self, gcode, save_name='lines_array'):
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
            # Move the laser off the surface of the bottle
            print('G0 Y{0:.4f}'.format(-self.focal_offset), file=text_file)
            # Begin with laser focus off the target surface so no ablation (drawing a line will refocus)
            print('G0 Y{0:.4f}'.format(-self.dy), file=text_file)
            # Write coordinates
            print(gcode, file=text_file)
            # End with laser focus on the target surface again
            print('G0 Y{0:.4f}\n'.format(self.dy), file=text_file)
            # Move the laser away from the bottle
            if self.final_offset:
                print('G0 Y{0:.4f}'.format(-self.final_offset), file=text_file)
            # End program
            print('M2', file=text_file)
