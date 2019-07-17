# Functions to build up the dataset.
# Authors: Andrea Maracani, Davide Talon. 2019


from scipy.io import wavfile
import features as f
import numpy as np
import math
import random
import librosa
import os
import shutil
import ntpath


def get_samples_from_noise(input_path, output_path, nOutput, seed, input_name='noise', output_name='noise', nInput=6,
                           duration=1):
    """From wav files of noise take random samples.
    :param input_path: directory path of input files.
    :param output_path: directory path of output samples.
    :param input_name: common name of input files (follow by increasing numbers).
    :param output_name: common name of output files (follow by increasing numbers).
    :param nInput: number of noise files (input)
    :param nOutput: number of output files
    :param duration: duration of each output file(in seconds)
    :param seed: seed for random number generator
    :returns: None.
    """

    # set the random seed
    random.seed(seed)

    # initialize array for input files
    input_data = np.empty(nInput, dtype=object)

    # initialize array for input sample rates
    sample_rates = np.empty(nInput)

    # get input data
    for i in range(nInput):
        sample_rates[i], input_data[i] = wavfile.read(input_path + input_name + str(i) + ".wav")

    # sample randomly input files and save output files
    for i in range(nOutput):
        # select a random file
        file_number = random.randrange(nInput)

        # evaluate the number of samples we need to take from that file to build a new file
        number_of_samples = int(duration * sample_rates[file_number])

        # starting sample
        sample_start = random.randrange(len(input_data[file_number]) - number_of_samples - 1)

        # output file = portion of input file of the user-defined duration
        file = input_data[file_number][sample_start:sample_start + number_of_samples]

        # save output file
        wavfile.write(output_path + output_name + str(i).zfill(len(str(nOutput))) + ".wav",
                      int(sample_rates[file_number]), file)


def delete_folder_for_each_class(input_path="data/", folder_name="augmentation/"):
    path, class_names, files = os.walk(input_path).__next__()

    class_names = sorted(class_names)

    class_paths = [input_path + cn + "/" for cn in class_names]

    # get number of classes
    number_of_classes = len(class_paths)

    # for every class
    for c in range(number_of_classes):
        shutil.rmtree(class_paths[c] + folder_name, ignore_errors=True)


def organize_dataset(input_path,
                     training_percentage=None,
                     validation_percentage=None,
                     test_percentage=None,
                     training_samples=None,
                     validation_samples=None,
                     test_samples=None,

                     # augmentation parameters
                     sample_rate=16000,
                     maximum_length=17000,
                     folder_name="augmentation/",
                     percentage=0.25,
                     pitch_change_min=-1,
                     pitch_change_max=1,
                     speed_change_min=0.95,
                     speed_change_max=1.05,
                     noise_max=0.005,

                     seed=1):
    path, class_names, files = os.walk(input_path).__next__()

    class_names = sorted(class_names)

    class_paths = [input_path + cn + "/" for cn in class_names]

    # get number of classes
    number_of_classes = len(class_paths)

    # for every class
    for c in range(number_of_classes):

        if not os.path.exists(class_paths[c] + "training/"):
            os.makedirs(class_paths[c] + "training/")
        if not os.path.exists(class_paths[c] + "validation/"):
            os.makedirs(class_paths[c] + "validation/")
        if not os.path.exists(class_paths[c] + "test/"):
            os.makedirs(class_paths[c] + "test/")

        # all available file paths for class c
        files = get_all_files(class_paths[c], exclude_directory_name=None)

        # number of available files for class c
        number_of_files = len(files)

        # number of files in the 3 sets
        training_size = training_samples
        validation_size = validation_samples
        test_size = test_samples

        # set sizes to 0 if they are specified in percentages
        if training_size is None:
            training_size = 0
        if validation_size is None:
            validation_size = 0
        if test_size is None:
            test_size = 0

        available_size = number_of_files - training_size - validation_size - test_size

        if training_samples is None:
            training_size = int(math.floor(available_size * training_percentage))
        if validation_samples is None:
            validation_size = int(math.floor(available_size * validation_percentage))
        if test_samples is None:
            test_size = int(math.floor(available_size * test_percentage))

        assert training_size + validation_size + test_size <= number_of_files, \
            "In class number " + str(c) + " the training size + validation size + test_size selected are greater then " \
                                          "available files..."

        # total number of files needed for class c
        number_of_files_needed = training_size + validation_size + test_size

        assert number_of_files >= number_of_files_needed, \
            "For class number " + str(c) + "we have less files then needed!"

        # permutation to divide randomly the files of this class in the three sets
        p = np.random.permutation(number_of_files)

        # fill the dataset with features taken from files
        for i in range(number_of_files_needed):

            file_name = ntpath.basename(files[p[i]])

            if i < training_size:

                shutil.move(files[p[i]], class_paths[c] + "training/" + file_name)

            elif i < training_size + validation_size:
                shutil.move(files[p[i]], class_paths[c] + "validation/" + file_name)
            else:
                shutil.move(files[p[i]], class_paths[c] + "test/" + file_name)

        augment_class(class_paths[c] + "training/",
                      sample_rate=sample_rate,
                      maximum_length=maximum_length,
                      folder_name=folder_name,
                      percentage=percentage,
                      pitch_change_min=pitch_change_min,
                      pitch_change_max=pitch_change_max,
                      speed_change_min=speed_change_min,
                      speed_change_max=speed_change_max,
                      noise_max=noise_max)


def augment_sample(data, sample_rate, maximum_length,
                   pitch_change_min=-1, pitch_change_max=1,
                   speed_change_min=0.95, speed_change_max=1.05,
                   noise_max=0.005):
    assert len(data) / speed_change_min <= maximum_length, \
        "with speed change parameters, the length of output exceed limit"

    # change pitch
    pitch_change = np.random.uniform(low=pitch_change_min, high=pitch_change_max)
    output = librosa.effects.pitch_shift(data, sample_rate, n_steps=pitch_change,
                                         bins_per_octave=12, res_type='kaiser_best')

    # change speed
    speed_change = np.random.uniform(low=speed_change_min, high=speed_change_max)
    output = librosa.effects.time_stretch(output, speed_change)

    # add noise
    noise_amp = noise_max * np.random.uniform() * np.max(data)
    output = data + noise_amp * np.random.normal(size=data.shape[0])

    return output


def augment_class(class_path, sample_rate=16000, maximum_length=17000,
                  folder_name="augmentation/", percentage=0.25,
                  pitch_change_min=-1, pitch_change_max=1,
                  speed_change_min=0.95, speed_change_max=1.05,
                  noise_max=0.005):
    # create folder if not exists
    if not os.path.exists(class_path + folder_name):
        os.makedirs(class_path + folder_name)

    # delete pre existing files in this directory
    delete_files_from_folder(class_path + folder_name)

    files = get_all_files(class_path, exclude_directory_name=folder_name)
    number_of_files = len(files)
    number_of_files_to_add = int(number_of_files * percentage)

    file_indices = np.random.randint(number_of_files, size=number_of_files_to_add)

    for i in range(len(file_indices)):
        file_to_aug = files[file_indices[i]]
        data, sample_rate = librosa.load(file_to_aug, sample_rate)
        data = augment_sample(data, sample_rate, maximum_length,
                              pitch_change_min=pitch_change_min, pitch_change_max=pitch_change_max,
                              speed_change_min=speed_change_min, speed_change_max=speed_change_max,
                              noise_max=noise_max)
        # save output file
        librosa.output.write_wav(class_path + folder_name + "aug_" + str(i).zfill(len(str(number_of_files_to_add))) +
                                 ".wav", data.astype("float32"), sample_rate, norm=True)


def augment_dataset(dataset_path, sample_rate=16000, maximum_length=17000,
                    folder_name="augmentation/", percentage=0.25,
                    pitch_change_min=-1, pitch_change_max=1,
                    speed_change_min=0.95, speed_change_max=1.05,
                    noise_max=0.005, seed=1):
    # set seed
    np.random.seed(seed)

    path, class_names, files = os.walk(dataset_path).__next__()

    class_paths = sorted([dataset_path + cn + "/" for cn in class_names])

    for c in class_paths:
        augment_class(c, sample_rate=sample_rate, maximum_length=maximum_length,
                      folder_name=folder_name, percentage=percentage,
                      pitch_change_min=pitch_change_min, pitch_change_max=pitch_change_max,
                      speed_change_min=speed_change_min, speed_change_max=speed_change_max,
                      noise_max=noise_max)


def create_dataset(input_path,
                   class_names=None,

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
                   shift_static=0, scale_static=1,
                   shift_delta=0, scale_delta=1,
                   shift_delta_delta=0, scale_delta_delta=1,

                   exclude_augmentation=False,
                   augmentation_folder="augmentation",

                   print_info=True):
    ################ SET CLASSES #######################################################################################

    if print_info:
        print("Analyzing classes and files...")

    # if class_names is set to None all classes are used
    if class_names is None:
        path, class_names, files = os.walk(input_path).__next__()

    class_names = sorted(class_names)

    class_paths = [input_path + cn + "/" for cn in class_names]

    # get number of classes
    number_of_classes = len(class_paths)

    if print_info:
        print("\n####################### CLASS INFOS #################################\n")
        print("Class#____Class Name____Training____Validation______Test_________Total")

    training_size = []
    validation_size = []
    test_size = []

    for c in range(number_of_classes):

        if not exclude_augmentation:
            augmentation_folder = None

        files_training = get_all_files(class_paths[c] + "training/", exclude_directory_name=augmentation_folder)
        files_validation = get_all_files(class_paths[c] + "validation/", exclude_directory_name=augmentation_folder)
        files_test = get_all_files(class_paths[c] + "test/", exclude_directory_name=augmentation_folder)

        # number of files in the 3 sets

        training_size.append(len(files_training))
        validation_size.append(len(files_validation))
        test_size.append(len(files_test))

        if print_info:
            print("  " + ("0" + str(c))[-2:] + "  " + "    " +
                  (class_names[c] + "       ")[0:10] + "    " +
                  "  " + ("00000" + str(training_size[c]))[-5:] + "      " +
                  "  " + ("00000" + str(validation_size[c]))[-5:] + "      " +
                  "  " + ("00000" + str(test_size[c]))[-5:] + "      " +
                  "  " + ("00000" + str(training_size[c] + validation_size[c] + test_size[c]))[-5:])

    # number of files in the 3 sets and total number of files
    training_size = sum(training_size)
    validation_size = sum(validation_size)
    test_size = sum(test_size)
    total_size = training_size + validation_size + test_size

    if print_info:
        print("\n###################### DATASET INFOS ################################\n")
        print("Training set has " + str(training_size) + " samples.")
        print("Validation set has " + str(validation_size) + " samples.")
        print("Test set has " + str(test_size) + " samples.")
        print("The dataset in total has " + str(total_size) + " samples.")
        print("\n#####################################################################\n")

    ##################### ALLOCATE MEMORY FOR DATASET ##################################################################

    # the shape of a feature image
    feature_shape = (target_frame_number, number_of_filters, 1)

    if add_delta:
        feature_shape = (target_frame_number, number_of_filters, 3)

    # initialize empty dataset
    training_x = np.empty((training_size,) + feature_shape)
    validation_x = np.empty((validation_size,) + feature_shape)
    test_x = np.empty((test_size,) + feature_shape)

    training_y = np.empty(training_size, dtype=int)
    validation_y = np.empty(validation_size, dtype=int)
    test_y = np.empty(test_size, dtype=int)

    ############# STARTING EFFECTIVE CREATION OF DATASET ###############################################################

    # permutations to make the order of the three sets pseudo-random
    training_permutation = np.random.permutation(training_size)
    validation_permutation = np.random.permutation(validation_size)
    test_permutation = np.random.permutation(test_size)

    # indices to fill sets
    i_tr = 0
    i_va = 0
    i_te = 0

    # number of files already processed
    files_processed = 0

    # current percentage of the process
    percentage = 0

    if print_info:
        print("Dataset creation is " + str(int(percentage)) + "% completed")

    # for every class
    for c in range(number_of_classes):

        # for every set (train, val, test)
        for set_number in range(3):

            if not exclude_augmentation:
                augmentation_folder = None

            if set_number == 0:
                files = get_all_files(class_paths[c] + "training/", exclude_directory_name=augmentation_folder)
            elif set_number == 1:
                files = get_all_files(class_paths[c] + "validation/", exclude_directory_name=augmentation_folder)
            elif set_number == 2:
                files = get_all_files(class_paths[c] + "test/", exclude_directory_name=augmentation_folder)

            for i in range(len(files)):
                # read a random file (with no repetition) among the available in class c
                fs, signal = wavfile.read(files[i])

                # get features from current file
                features = f.get_time_padded_features(signal, sample_rate=fs,
                                                      # PADDING
                                                      target_frame_number=target_frame_number,
                                                      random_time_shift=random_time_shift,
                                                      smooth=smooth,
                                                      smooth_length=smooth_length,

                                                      pre_emphasis_coef=pre_emphasis_coef,
                                                      # FRAMING PARAMETERS
                                                      frame_length=frame_length,
                                                      frame_step=frame_step,
                                                      window_function=window_function,

                                                      # MEL FILTERS PARAMETERS
                                                      hertz_from=hertz_from,
                                                      hertz_to=hertz_to,
                                                      number_of_filters=number_of_filters,

                                                      # FFT PARAMETERS
                                                      power_of_2=power_of_2,

                                                      # OUTPUT SETTINGS
                                                      dtype=dtype,
                                                      use_dct=use_dct,
                                                      add_delta=add_delta,

                                                      # NORMALIZATION
                                                      shift_static=shift_static,
                                                      scale_static=scale_static,
                                                      shift_delta=shift_delta,
                                                      scale_delta=scale_delta,
                                                      shift_delta_delta=shift_delta_delta,
                                                      scale_delta_delta=scale_delta_delta)

                if set_number == 0:

                    training_x[training_permutation[i_tr], ] = features
                    training_y[training_permutation[i_tr]] = c
                    i_tr = i_tr + 1

                elif set_number == 1:

                    validation_x[validation_permutation[i_va], ] = features
                    validation_y[validation_permutation[i_va]] = c
                    i_va = i_va + 1

                elif set_number == 2:
                    test_x[test_permutation[i_te], ] = features
                    test_y[test_permutation[i_te]] = c
                    i_te = i_te + 1


                files_processed = files_processed + 1

                if int(files_processed * 100 / total_size) > percentage and print_info:
                    percentage = int(files_processed * 100 / total_size)
                    print("Dataset creation is " + str(int(percentage)) + "% completed")

    return training_x, validation_x, test_x, training_y, validation_y, test_y


# def create_dataset_and_split(input_path,
#                              class_names=None,
#                              training_percentage=None,
#                              validation_percentage=None,
#                              test_percentage=None,
#                              training_samples=None,
#                              validation_samples=None,
#                              test_samples=None,
#
#                              pre_emphasis_coef=0.95,
#                              frame_length=400,
#                              frame_step=160,
#                              window_function=np.hamming,
#                              target_frame_number=110,
#                              random_time_shift=True,
#                              smooth=True,
#                              smooth_length=5,
#
#                              hertz_from=300,
#                              hertz_to=None,
#                              number_of_filters=40,
#                              power_of_2=True,
#                              dtype='float32',
#                              use_dct=False,
#                              add_delta=True,
#
#                             # NORMALIZATION
#                              shift_static=0, scale_static=1,
#                              shift_delta=0, scale_delta=1,
#                              shift_delta_delta=0, scale_delta_delta=1,
#
#                              exclude_augmentation=False,
#                              augmentation_folder="augmentation",
#
#                              print_info=True):
#     ################# CHECK PARAMETERS #################################################################################
#
#     # check that either percentages are set or number of samples
#     assert (training_percentage is None) + (training_samples is None) == 1, \
#         "One and only one among training percentage and training samples must be set!"
#     assert (validation_percentage is None) + (validation_samples is None) == 1, \
#         "One and only one among validation percentage and validation samples must be set!"
#     assert (test_percentage is None) + (test_samples is None) == 1, \
#         "One and only one among test percentage and test samples must be set!"
#
#     # check split percentages are set correctly
#     if (training_percentage is not None) + (validation_percentage is not None) + (test_percentage is not None) == 3:
#         assert training_percentage >= 0 and validation_percentage >= 0 and test_percentage >= 0, \
#             "Split percentages must be positive!"
#
#         assert training_percentage + validation_percentage + test_percentage <= 1, \
#             "Split percentages sum can't be greater then 1!"
#
#     elif (training_percentage is not None) + (validation_percentage is not None) == 2:
#         assert training_percentage >= 0 and validation_percentage >= 0, "Split percentages must be positive!"
#         assert training_percentage + validation_percentage <= 1, "Split percentages sum is greater then 1!"
#
#     elif (training_percentage is not None) + (test_percentage is not None) == 2:
#         assert training_percentage >= 0 and test_percentage >= 0, "Split percentages must be positive!"
#         assert training_percentage + test_percentage <= 1, "Split percentages sum is greater then 1!"
#
#     else:
#         assert (validation_percentage >= 0) and (test_percentage >= 0), "Split percentages must be positive!"
#         assert validation_percentage + test_percentage <= 1, "Split percentages sum is greater then 1!"
#
#     if print_info:
#         print("Parameters check completed, the creation of the dataset is starting...")
#
#     ################ SET CLASSES #######################################################################################
#
#     if print_info:
#         print("Analyzing classes and files...")
#
#     # if class_names is set to None all classes are used
#     if class_names is None:
#         path, class_names, files = os.walk(input_path).__next__()
#
#     class_names = sorted(class_names)
#
#     class_paths = [input_path + cn + "/" for cn in class_names]
#
#     # get number of classes
#     number_of_classes = len(class_paths)
#
#     # composition of the dataset
#     composition = []
#
#     if print_info:
#         print("\n####################### CLASS INFOS #################################\n")
#         print("Class#____Class Name____Training____Validation______Test_________Total")
#
#     for c in range(number_of_classes):
#
#         if not exclude_augmentation:
#             augmentation_folder = None
#
#         files = get_all_files(class_paths[c], exclude_directory_name=augmentation_folder)
#
#         # number of files in class c
#         number_of_files = len(files)
#
#         # number of files in the 3 sets
#
#         training_size = training_samples
#         validation_size = validation_samples
#         test_size = test_samples
#
#         # set sizes to 0 if they are specified in percentages
#         if training_size is None:
#             training_size = 0
#         if validation_size is None:
#             validation_size = 0
#         if test_size is None:
#             test_size = 0
#
#         available_size = number_of_files - training_size - validation_size - test_size
#
#         if training_samples is None:
#             training_size = int(math.floor(available_size * training_percentage))
#         if validation_samples is None:
#             validation_size = int(math.floor(available_size * validation_percentage))
#         if test_samples is None:
#             test_size = int(math.floor(available_size * test_percentage))
#
#         assert training_size + validation_size + test_size <= number_of_files, \
#             "In class number " + str(c) + " the training size + validation size + test_size selected are greater then " \
#                                           "available files..."
#
#         if print_info:
#             print("  " + ("0" + str(c))[-2:] + "  " + "    " +
#                   (class_names[c] + "       ")[0:10] + "    " +
#                   "  " + ("00000" + str(training_size))[-5:] + "      " +
#                   "  " + ("00000" + str(validation_size))[-5:] + "      " +
#                   "  " + ("00000" + str(test_size))[-5:] + "      " +
#                   "  " + ("00000" + str(training_size + validation_size + test_size))[-5:])
#
#         # append to composition
#         composition.append([training_size, validation_size, test_size])
#
#     # transform composition into a numpy array
#     composition = np.array(composition)
#
#     # number of files in the 3 sets and total number of files
#     training_size = sum(composition[:, 0])
#     validation_size = sum(composition[:, 1])
#     test_size = sum(composition[:, 2])
#     total_size = training_size + validation_size + test_size
#
#     if print_info:
#         print("\n###################### DATASET INFOS ################################\n")
#         print("Training set has " + str(training_size) + " samples.")
#         print("Validation set has " + str(validation_size) + " samples.")
#         print("Test set has " + str(test_size) + " samples.")
#         print("The dataset in total has " + str(total_size) + " samples.")
#         print("\n#####################################################################\n")
#
#     ##################### ALLOCATE MEMORY FOR DATASET ##################################################################
#
#     # the shape of a feature image
#     feature_shape = (target_frame_number, number_of_filters, 1)
#
#     if add_delta:
#         feature_shape = (target_frame_number, number_of_filters, 3)
#
#     # initialize empty dataset
#     training_x = np.empty((training_size,) + feature_shape)
#     validation_x = np.empty((validation_size,) + feature_shape)
#     test_x = np.empty((test_size,) + feature_shape)
#
#     training_y = np.empty(training_size, dtype=int)
#     validation_y = np.empty(validation_size, dtype=int)
#     test_y = np.empty(test_size, dtype=int)
#
#
#     ############# STARTING EFFECTIVE CREATION OF DATASET ###############################################################
#
#     # permutations to make the order of the three sets pseudo-random
#     training_permutation = np.random.permutation(training_size)
#     validation_permutation = np.random.permutation(validation_size)
#     test_permutation = np.random.permutation(test_size)
#
#     # tmp indices to iterate over the sets
#     i_tr = 0
#     i_va = 0
#     i_te = 0
#
#     # number of files already processed
#     files_processed = 0
#
#     # current percentage of the process
#     percentage = 0
#
#     if print_info:
#         print("Dataset creation is " + str(int(percentage)) + "% completed")
#
#     # for every class
#     for c in range(number_of_classes):
#
#         if not exclude_augmentation:
#             augmentation_folder = None
#
#         # all available file paths for class c
#         files = get_all_files(class_paths[c], exclude_directory_name=augmentation_folder)
#
#         # number of available files for class c
#         number_of_files_available = len(files)
#
#         # number of files for each set for class c
#         n_tr = composition[c, 0]
#         n_va = composition[c, 1]
#         n_te = composition[c, 2]
#
#         # total number of files needed for class c
#         number_of_files_needed = n_tr + n_va + n_te
#
#         assert number_of_files_available >= number_of_files_needed, \
#             "For class number " + str(c) + "we have less files then needed!"
#
#         # permutation to divide randomly the files of this class in the three sets
#         p = np.random.permutation(number_of_files_available)
#
#         # fill the dataset with features taken from files
#         for i in range(number_of_files_needed):
#
#             # read a random file (with no repetition) among the available in class c
#             fs, signal = wavfile.read(files[p[i]])
#
#             # get features from current file
#             features = f.get_time_padded_features(signal, sample_rate=fs,
#                                                   # PADDING
#                                                   target_frame_number=target_frame_number,
#                                                   random_time_shift=random_time_shift,
#                                                   smooth=smooth,
#                                                   smooth_length=smooth_length,
#
#                                                   pre_emphasis_coef=pre_emphasis_coef,
#                                                   # FRAMING PARAMETERS
#                                                   frame_length=frame_length,
#                                                   frame_step=frame_step,
#                                                   window_function=window_function,
#
#                                                   # MEL FILTERS PARAMETERS
#                                                   hertz_from=hertz_from,
#                                                   hertz_to=hertz_to,
#                                                   number_of_filters=number_of_filters,
#
#                                                   # FFT PARAMETERS
#                                                   power_of_2=power_of_2,
#
#                                                   # OUTPUT SETTINGS
#                                                   dtype=dtype,
#                                                   use_dct=use_dct,
#                                                   add_delta=add_delta,
#
#                                                   # NORMALIZATION
#                                                   shift_static=shift_static,
#                                                   scale_static=scale_static,
#                                                   shift_delta=shift_delta,
#                                                   scale_delta=scale_delta,
#                                                   shift_delta_delta=shift_delta_delta,
#                                                   scale_delta_delta=scale_delta_delta)
#
#             # add the first n_tr files (chosen randomly) in the training set (in a random position)
#             if i < n_tr:
#                 training_x[training_permutation[i_tr], ] = features
#                 training_y[training_permutation[i_tr]] = c
#                 i_tr = i_tr + 1
#             # add  n_va files (chosen randomly) in the validation set (in a random position)
#             elif i < n_tr + n_va:
#                 validation_x[validation_permutation[i_va],] = features
#                 validation_y[validation_permutation[i_va]] = c
#                 i_va = i_va + 1
#             # add  n_te files (chosen randomly) in the test set (in a random position)
#             else:
#                 test_x[test_permutation[i_te], ] = features
#                 test_y[test_permutation[i_te]] = c
#                 i_te = i_te + 1
#
#             # increase counter of files processed
#             files_processed = files_processed + 1
#
#             if int(files_processed * 100 / total_size) > percentage and print_info:
#                 percentage = int(files_processed * 100 / total_size)
#                 print("Dataset creation is " + str(int(percentage)) + "% completed")
#
#
#     return training_x, validation_x, test_x, training_y, validation_y, test_y


def delete_files_from_folder(input_path):
    """Delete all files in the specified directory.
    :param input_path: the path of the directory.
    """
    for the_file in os.listdir(input_path):
        file_path = os.path.join(input_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_all_files(input_path, exclude_directory_name=None):
    """Starting from the directory dir_path it writes into a list all the paths of all the files inside dir_path or
    inside one subdirectory.
    :param input_path: the path of the directory.
    :param exclude_directory_name: name of a directory to exclude in the search.
    :return: a list of paths (one for each file in dir_path or one of its subdirectories)
    """

    directories = [input_path]
    files_list = []

    while len(directories) > 0:
        current_dir = directories.pop()

        path, dirs, files = os.walk(current_dir).__next__()

        if exclude_directory_name is not None and exclude_directory_name in dirs:
            dirs.remove(exclude_directory_name)

        for i in range(len(dirs)):
            dirs[i] = current_dir + dirs[i] + "/"

        for i in range(len(files)):
            files[i] = current_dir + files[i]

        if len(files) > 0:
            files_list.extend(files)

        if len(dirs) > 0:
            directories.extend(dirs)

    return files_list

# def pad_signal_with_noise(signal, noise, samples_before, samples_after, overlapping_before, overlapping_after,
#                           intensity):
#     noise_before = noise[0:samples_before]
#     noise_before_overlap = noise[samples_before:samples_before + overlapping_before]
#     noise_after_overlap = noise[-(samples_after + overlapping_after):-samples_after]
#     noise_after = noise[-samples_after:]
#
#     print(len(noise_before))
#     print(len(noise_before_overlap))
#     print(len(noise_after_overlap))
#     print(len(noise_after))
#     # apply ramp from 0 to intensity level
#
#     noise_before = [noise_before[i] * (intensity * i / samples_before) for i in range(samples_before)]
#     noise_before_overlap = [noise_before_overlap[i] * (intensity * (1 - i / overlapping_before)) for i in
#                             range(overlapping_before)]
#     noise_after_overlap = [noise_after_overlap[i] * (intensity * i / overlapping_after) for i in
#                            range(overlapping_after)]
#     noise_after = [noise_after[i] * (intensity * (1 - i / samples_after)) for i in
#                    range(samples_after)]
#     signal_ = np.copy(signal)
#     signal_[0:overlapping_before] = [noise_before_overlap[i] + (signal[i] * i / overlapping_before) for i in
#                                     range(overlapping_before)]
#     signal_[-overlapping_after:] = [noise_after_overlap[i] + signal[i] * (1 - i / overlapping_after) for i in
#                                   range(overlapping_after)]
#
#     print("sign_length ", len(signal_))
#     signal_ = np.array(signal_)
#     return np.concatenate((noise_before, signal_, noise_after), axis=0)
