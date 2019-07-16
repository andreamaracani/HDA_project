import util as u
import numpy as np

# full dataset parameters
mean_static = 5.894829478708612
mean_delta = 0.011219158052764553
mean_delta2 = -0.0011225424307293912
std_static = 6.498562130522682
std_delta = 0.6461172437961028
std_delta2 = 0.24652535283769192
max_static = 25.442123413085938
min_static = -36.04365158081055
max_delta = 16.69818878173828
min_delta = -15.686269760131836
max_delta2 = 9.076530456542969
min_delta2 = -8.840045928955078

# for normalization
shift_static = -min_static
scale_static = 1 / (max_static - min_static)
shift_delta = -min_delta
scale_delta = 1 / (max_delta - min_delta)
shift_delta_delta = -min_delta2
scale_delta_delta = 1 / (max_delta2 - min_delta2)

# for standardisation
# shift_static = -mean_static
# scale_static = 1/std_static
# shift_delta = -mean_delta
# scale_delta = 1/std_delta
# shift_delta_delta = -mean_delta2
# scale_delta_delta = 1/std_delta2

# for no transformation
# shift_static = 0
# scale_static = 1
# shift_delta = 0
# scale_delta = 1
# shift_delta_delta = 0
# scale_delta_delta = 1

input_path = "data/"
class_names = ["00 zero/", "01 one/"]  # set to None to use all classes

training_x, validation_x, test_x, training_y, validation_y, test_y = \
    u.create_dataset_and_split(input_path,
                               class_names=class_names,
                               training_percentage=1,
                               validation_percentage=0.0,
                               test_percentage=None,
                               training_samples=None,
                               validation_samples=None,
                               test_samples=0,

                               pre_emphasis_coef=0.95,
                               frame_length=400,
                               frame_step=160,
                               window_function=np.hamming,
                               target_frame_number=110,
                               random_time_shift=True,
                               smooth=True,
                               smooth_length=5,

                               hertz_from=300,
                               hertz_to=None,
                               number_of_filters=40,
                               power_of_2=True,
                               dtype='float32',
                               use_dct=False,
                               add_delta=True,

                               # NORMALIZATION
                               shift_static=shift_static,
                               scale_static=scale_static,
                               shift_delta=shift_delta,
                               scale_delta=scale_delta,
                               shift_delta_delta=shift_delta_delta,
                               scale_delta_delta=scale_delta_delta,

                               exclude_augmentation=True,
                               augmentation_folder="augmentation",

                               print_info=True)
