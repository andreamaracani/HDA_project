# Functions to build up the dataset.
# Authors: Andrea Maracani, Davide Talon. 2019


from scipy.io import wavfile
import features as f
import numpy as np
import random
import os


def get_samples_from_noise(input_path, output_path, nOutput, seed, input_name='noise', output_name='noise', nInput=6, duration=1):
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


def create_dataset(input_path, max_files_per_class=None, save=False, printInfo=True):
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


def split_dataset(dataset, n_samples_test, training_percentage):
    """Divide the dataset into training, validation and test set
     :param dataset: a numpy array representing the dataset.
     :param n_samples_test: the number of samples, for each class, in the test set.
     :param training_percentage: percentage of remaining samples in the training set
     :return: the three sets randomly permuted with corresponding three array of labels
     """

    assert 0 < training_percentage and  training_percentage < 1

    # assert that there are enough samples for every class
    for i in range(dataset.shape[0]):
        assert len(dataset[i]) > n_samples_test

    training = []
    validation = []
    test = []
    labels_training = []
    labels_validation = []
    labels_test = []

    # for every class
    for i in range(dataset.shape[0]):
        # number of samples present in class i
        n_samples = len(dataset[i])#.shape[0]

        n_samples_val = int((n_samples - n_samples_test)*(1-training_percentage))
        n_samples_train = n_samples-n_samples_val-n_samples_test

        p = np.random.permutation(list(range(n_samples)))

        train_indices = p[0:n_samples_train]
        val_indices = p[n_samples_train: n_samples_train+n_samples_val]
        test_indices = p[n_samples_train+n_samples_val:]

        # if different classes have different number of samples, then dataset[i] is a simple list
        # and we need to put it inside a numpy array
        current = np.array(dataset[i])

        training.extend(current[train_indices])
        validation.extend(current[val_indices])
        test.extend(current[test_indices])

        labels_training.extend([i for j in range(n_samples_train)])
        labels_validation.extend([i for j in range(n_samples_val)])
        labels_test.extend([i for j in range(n_samples_test)])
        


    # training, validation, test = np.array(training), np.array(validation), np.array(test)
    training = np.stack( training, axis=0)
    validation = np.stack(validation, axis=0)
    test = np.stack(test, axis=0)


    # labels_training, labels_validation, labels_test = np.array(labels_training), np.array(labels_validation),\
    #                                                   np.array(labels_test)
    
    labels_training = np.stack(labels_training, axis=0)
    labels_validation = np.stack(labels_validation, axis=0)
    labels_test = np.stack(labels_test, axis=0)

    training, labels_training = shuffle_dataset(training, labels_training)
    validation, labels_validation = shuffle_dataset(validation, labels_validation)
    test, labels_test = shuffle_dataset(test, labels_test)

    return training, validation, test, labels_training, labels_validation, labels_test

#NOT USED....
# def complementary_indices(indices, number_of_indices, min_index=0):
#     """Computes the list of all indices between min and maz that are not present in indices
#     :param indices: a list of indices.
#     :param number_of_indices: the number of indices (the number of origianl indices + number of complementary indices).
#     :param min_index: the minumum index.
#     :return: a list of indices (complementary to indices list)
#     """
#
#     comp = set(range(min_index,min_index+number_of_indices))
#     indices = set(indices)
#
#     return list(comp-indices)


def shuffle_dataset(dataset, labels):
    """Shuffle the dataset and labels numpy array with the same permutation
    :param dataset: first array to permute.
    :param labels: second array to permute.
    :return: the two array randomly permuted (but in the same way)
    """
    p = np.random.permutation(len(labels))
    return dataset[p], labels[p]

