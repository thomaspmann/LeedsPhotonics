"""
Scripts for implanting smoke on angled bottle.
"""

from Ultramatis.array_to_gcode_module.array_scripts import qr_array, square
from Ultramatis.array_to_gcode_module.main import ArrayToGcode


def lay_smoke_square():
    """Draw black square on the target to lay a layer of smoke on the bottle, which will then be etched"""
    fname = '../square_smoke'
    array = square()
    # Write to G code
    gcode = ArrayToGcode(array,
                         pixel_width=15,
                         lines_per_pixel=86,
                         speed=3.0,
                         d_focus=1.0,
                         spot_size=0,
                         focal_offset=0.0,
                         final_offset=5.0,
                         degrees=10)
    gcode.print_size()
    gcode.to_file(save_name=fname)


def angled_qr():
    """Draw QR code on the bottle at an angle"""
    fname = '../qr_angled'
    array = qr_array()
    # Write to G code
    gcode = ArrayToGcode(array,
                         pixel_width=0.7,
                         lines_per_pixel=6,
                         speed=3.0,
                         d_focus=-1.0,
                         spot_size=0,
                         focal_offset=0.0,
                         final_offset=5.0,
                         degrees=10)
    gcode.print_size()
    gcode.to_file(save_name=fname)


if __name__ == "__main__":
    # lay_smoke_square()
    angled_qr()

    # Online G-code viewer: http://nraynaud.github.io/webgcode/
