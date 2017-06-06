from Ultramatis.array_to_gcode_module.array_scripts import qr_array
from Ultramatis.array_to_gcode_module.main import ArrayToGcode


def flat(fname, array):
    """Write array onto a surface and save as fname"""

    # Write to G code
    gcode = ArrayToGcode(array,
                         pixel_width=0.7,
                         lines_per_pixel=6,
                         speed=5.0,
                         d_focus=-2,
                         spot_size=0.2,
                         focal_offset=0.8,
                         degrees=20)
    gcode.print_size()
    # gcode.invert()

    gcode.plot_array(save_name=fname, show=True)
    gcode.to_file(save_name=fname)

if __name__ == "__main__":
    # fname = '../alphanumeric/14'
    # array = alphanumeric()

    fname = '../QR_code_angled'
    array = qr_array()

    # fname = '../test_array_dy_positive'
    # array = test_array()

    # fname = '../square'
    # array = square()

    flat(fname, array)
    # angled.py(fname, array)

    # Online G-code viewer: http://nraynaud.github.io/webgcode/
