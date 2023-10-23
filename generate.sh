rm comp/*
mkdir comp
g++ my_functions.cpp -g -std=c++17 -I ./Halide/include -L ./Halide/lib -lHalide -lpthread -ldl -o generate_my_functions
echo "Creating generate..." >&2
g++ test.cpp Halide/share/Halide/tools/GenGen.cpp -g -std=c++17 -fno-rtti -I Halide/include -L Halide/lib -lHalide -lpthread -ldl -o generate
echo "Generating matmul..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g matmul -f halide_matmul_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g matmul -f halide_matmul -e static_library,h target=host
echo "Generating adjust_brightness schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_brightness -f halide_adjust_brightness_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_brightness -f halide_adjust_brightness -e static_library,h target=host
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_brightness_bw -f halide_adjust_brightness_bw_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_brightness_bw -f halide_adjust_brightness_bw -e static_library,h target=host
echo "Generating invert schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g invert -f halide_invert_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g invert -f halide_invert -e static_library,h target=host
echo "Generating rgb_to_grayscale schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g rgb_to_grayscale -f halide_rgb_to_grayscale_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g rgb_to_grayscale -f halide_rgb_to_grayscale -e static_library,h target=host
echo "Generating solarize schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g solarize -f halide_solarize_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g solarize -f halide_solarize -e static_library,h target=host
echo "Generating adjust_gamma schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_gamma -f halide_adjust_gamma_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_gamma -f halide_adjust_gamma -e static_library,h target=host
echo "Generating adjust_saturation schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_saturation -f halide_adjust_saturation_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_saturation -f halide_adjust_saturation -e static_library,h target=host
echo "Generating adjust_contrast schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_contrast -f halide_adjust_contrast_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_contrast -f halide_adjust_contrast -e static_library,h target=host
echo "Generating autocontrast schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g autocontrast -f halide_autocontrast_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g autocontrast -f halide_autocontrast -e static_library,h target=host
echo "Generating normalize schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g normalize -f halide_normalize_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g normalize -f halide_normalize -e static_library,h target=host
echo "Generating posterize schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g posterize -f halide_posterize_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g posterize -f halide_posterize -e static_library,h target=host
echo "Generating adjust_hue schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_hue -f halide_adjust_hue_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_hue -f halide_adjust_hue -e static_library,h target=host
echo "Generating adjust_sharpness schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_sharpness -f halide_adjust_sharpness_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_sharpness -f halide_adjust_sharpness -e static_library,h target=host
echo "Generating elastic_transform schedule..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g elastic_transform -f halide_elastic_transform_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019 autoscheduler.beam_size=128
LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g elastic_transform -f halide_elastic_transform -e static_library,h target=host


echo "Finished schedules, cleaning up..." >&2
LD_LIBRARY_PATH=./Halide/lib ./generate_my_functions
rm -rf generate_my_functions
rm -rf generate


#LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_brightness -f adjust_brightness_auto -e static_library,h,schedule -p /libautoschedule_adams2019 target=host autoscheduler=Adams2019
#LD_LIBRARY_PATH=./Halide/lib ./generate -o comp/ -g adjust_brightness -f adjust_brightness -e static_library,h target=host

# /libautoschedule_adams2019 target=host autoscheduler=Adams2019
# /libautoschedule_li2018 target=host autoscheduler=Li2018
# /libautoschedule_mullapudi2016 target=host autoscheduler=Mullapudi2016