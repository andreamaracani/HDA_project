# Functions to build up the dataset.
# Authors: Andrea Maracani, Davide Talon. 2019


from scipy.io import wavfile
import features as f
import numpy as np
import random
import os


def get_samples_from_noise(input_path, output_path, input_name, output_name, nInput, nOutput, duration, seed):
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
        sample_start = random.randrange(len(input_data[file_number])-number_of_samples-1)

        # output file = portion of input file of the user-defined duration
        file = input_data[file_number][sample_start:sample_start+number_of_samples]

        # save output file
        wavfile.write(output_path+output_name+str(i).zfill(len(str(nOutput)))+".wav", int(sample_rates[file_number]), file)


def create_dataset(input_path, max_files_per_class = None, save = True, printInfo = True):
    """From the dataset input path it build up a big numpy array with all data.
    :param input_path: directory path of the dataset.
    :param max_files_per_class: maximum number of files for each class. If it is set to None we include all files
    :param save. Save to disk a file for every class
    :param printInfo: if it is true it prints infos about the dataset
    :return: a numpy array containing the organized dataset
    """

    # get the path of every class directory of the dataset
    classes = get_class_dirs(input_path)

    # number of all files in the dataset (just to estimate processing time)
    number_of_all_files = get_number_of_files(input_path)

    # initialize empty dataset
    dataset = []

    # number of files already processed (just to estimate processing time)
    files_processed = 0

    # current percentage of the process
    percentage = 0

    if printInfo:
        print("Starting creation of dataset...")
        print("Dataset creation is " + str(int(percentage)) + "% completed")

    # for every class
    for c in range(len(classes)):
        files = get_all_files(classes[c])

        nFiles = len(files)

        if max_files_per_class is not None:
            nFiles = min(max_files_per_class, nFiles)

        list_of_features = []
        for i in range(nFiles):
            fs, data = wavfile.read(files[i])
            features = f.get_features(data, fs, window_function=np.hamming, number_of_filters=40)
            list_of_features.append(features)

            # increase counter of files processed
            files_processed = files_processed + 1

            if int(files_processed*100/number_of_all_files) > percentage and printInfo:
                percentage = int(files_processed*100/number_of_all_files)
                print("Dataset creation is " + str(int(percentage)) + "% completed")

        if save:
            if printInfo:
                print("Saving file for class " + str(c))

            np.save("class" + str(c) + ".npy", np.array(list_of_features))

        dataset.append(list_of_features)

    # end for

    if printInfo:
        print(get_dataset_info(input_path))

    return np.array(dataset)


def get_class_dirs(input_path):
    """From the dataset input path it finds the path of all class folders
    :param input_path: directory path of the dataset.
    """

    path, dirs, files = os.walk(input_path).__next__()

    for i in range(len(dirs)):
        dirs[i] = input_path + dirs[i] + "/"

    return dirs


def get_number_of_files(input_path):
    """Count the number of files in dir_path directory and in all subdirectories
        :param input_path: the path of the directory
        :return: the number of files in dir_path and in its subdirectories
        """
    return sum([len(files) for path, dirs, files in os.walk(input_path)])


def get_dataset_info(input_path):
    """Get info about the classes and files distribution in the dataser
           :param input_path: the path of the directory
           :return: a info string
           """
    classes = get_class_dirs(input_path)

    number_of_classes = len(classes)
    number_of_files = get_number_of_files(input_path)

    info = "In the dataset there are " + str(number_of_files) + " files divided in " + str(number_of_classes) + " classes \n"

    for i in range(len(classes)):
        info = info + "Class " + str(i) + " has " + str(get_number_of_files(classes[i])) + " samples. \n"

    return info


def get_all_files(input_path):
    """Starting from the directory dir_path it writes into a list all the paths of all the files inside dir_path or
    inside one subdirectory.
    :param input_path: the path of the directory
    :return: a list of paths (one for each file in dir_path or one of its subdirectories)
    """

    directories = [input_path]
    files_list = []

    while len(directories) > 0:
        current_dir = directories.pop()

        path, dirs, files = os.walk(current_dir).__next__()

        for i in range(len(dirs)):
            dirs[i] = current_dir + dirs[i] + "/"

        for i in range(len(files)):
            files[i] = current_dir + files[i]

        if len(files) > 0:
            files_list.extend(files)

        if len(dirs) > 0:
            directories.extend(dirs)

    return files_list
