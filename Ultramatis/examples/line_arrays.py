from Ultramatis.array_to_gcode_module.main import ArrayOfLines


def lines_array():
    st = ArrayOfLines(nlines=1,
                      speed=1.0,
                      dx=1.0,
                      dy=3.0,
                      dz=10.0,
                      focal_offset=0.8,
                      final_offset=-0.8)
    gcode = st.array()
    st.save(gcode, save_name='lines_array')


def sweep_height():
    st = ArrayOfLines(nlines=15,
                      speed=1.0,
                      dx=1.0,
                      dy=3.0,
                      dz=10.0,
                      focal_offset=0.0,
                      final_offset=0.0)
    gcode = st.sweep_height(dy_inc=0.1)
    st.save(gcode, save_name='sweep_height')


def sweep_speed():
    st = ArrayOfLines(nlines=15,
                      speed=15.0,
                      dx=1.0,
                      dy=3.0,
                      dz=10.0,
                      focal_offset=0.8,
                      final_offset=-0.8)
    gcode = st.sweep_speed(speed_inc=1.0)
    st.save(gcode, save_name='sweep_speedb')


if __name__ == "__main__":
    lines_array()
    # sweep_height()
    # sweep_speed()

    # Online G-code viewer: http://nraynaud.github.io/webgcode/
