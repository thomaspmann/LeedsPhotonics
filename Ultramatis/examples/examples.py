from Ultramatis.array_to_gcode_module.array_scripts import test_array, qr_array
from Ultramatis.array_to_gcode_module.main import ArrayToGcode, ArrayToGcodeAngled


def flat():
    # Make Array
    array = qr_array()
    # array = test_array()

    # Write to G code
    gcode = ArrayToGcode(array,
                         pixel_width=0.7,
                         lines_per_pixel=6,
                         speed=1.0,
                         dy=-3,
                         spot_size=0)
    gcode.print_size()
    # gcode.invert()
    gcode.plot_array()
    gcode.to_file(save_name='../QR_code_dy_negative')
    # gcode.to_file(save_name='../test_array_dy_positive')


def angled():
    # Make Array
    # array = qr_array()
    array = test_array()

    # Write to G code
    gcode = ArrayToGcodeAngled(degrees=75,
                               array=array,
                               pixel_width=0.5,
                               lines_per_pixel=6,
                               speed=1.0,
                               spot_size=0)
    gcode.print_size()
    # gcode.invert()
    gcode.plot_array()
    # gcode.to_file(save_name='QR_code_1mm_per_s')
    gcode.to_file(save_name='test_array_angled')


if __name__ == "__main__":
    flat()
    # angled()
    # Online G-code viewer: http://nraynaud.github.io/webgcode/
