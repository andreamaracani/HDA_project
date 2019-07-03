from scipy.io import wavfile
import numpy as np
import random
import os

def get_samples_from_noise(input_path, output_path, input_name, output_name, nFiles, nSamples, duration, seed):
    """From wav files of noise take random samples.
    :param input_path: directory path of input files.
    :param output_path: directory path of output samples.
    :param input_name: common name of input files (follow by increasing numbers).
    :param output_name: common name of output files (follow by increasing numbers).
    :param nFiles: number of noise files (input)
    :param nSamples: number of output files
    :param duration: duration of each output file(in seconds)
    :param seed: seed for random number generetor
    :returns: None.
    """

    # set the random seed
    random.seed(seed)

    # initialize array for input files
    input_data = np.empty(nFiles, dtype=object)

    # initialize array for input sample rates
    sample_rates = np.empty(nFiles)

    # get input data
    for i in range(nFiles):
        sample_rates[i], input_data[i] = wavfile.read(input_path + input_name + str(i) + ".wav")

    # sample randomly input files and save output files
    for i in range(nSamples):

        # select a random file
        file_number = random.randrange(nFiles)

        # evaluate the number of samples we need to take from that file to build a new file
        number_of_samples = int(duration * sample_rates[file_number])

        # starting sample
        sample_start = random.randrange(len(input_data[file_number]-number_of_samples-1))

        # output file = portion of input file of the user-defined duration
        file = input_data[file_number][sample_start:sample_start+number_of_samples]

        # save output file
        wavfile.write(output_path+output_name+str(i)+".wav", int(sample_rates[file_number]), file)


def get_class_dirs(input_path):
    """From the dataset input path it finds the path of all class folders
    :param input_path: directory path of the dataset.
    """

    path, dirs, files = os.walk(input_path).__next__()

    for i in range(len(dirs)):
        dirs[i] = input_path + dirs[i] + "/"

    return dirs

def get_all_files(dir_path):
    """Starting from the directory dir_path it writes into a list all the paths of all the files inside dir_path or
    inside one subdirectory.
    :param dir_path: the path of the directory
    :return: a list of paths (one for each file in dir_path or one of its subdirectories)
    """

    directories = [dir_path]
    files_list = []

    while len(directories)>0:
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
