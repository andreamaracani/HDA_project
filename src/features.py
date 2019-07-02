# Functions to extract filterbank features.
# Authors: Andrea Maracani, Davide Talon. 2019
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.fftpack import dct

# use true for debug
DEBUG = False


def pre_emphasis(signal, coef=0.95):
    """Pre emphasis FIR filter to enhance high frequencies.
    :param signal: the input signal
    :param coef: The pre emphasis coefficient (0 = no filter)
    :returns: filtered signal.
    """
    return np.append(signal[0], signal[1:] - signal[:-1] * coef)


def frame_signal(signal, sample_rate, frame_duration=0.025, frame_step=0.010, window_function=lambda z: np.ones((z,))):
    """Divide a signal into frames.
    :param signal: the array to be framed.
    :param sample_rate: the sample rate of the original audio signal.
    :param frame_duration: the duration (in seconds) of each frame.
    :param frame_step: the time distance between the beginning of two adjacent frames.
    :param window_function: a function that specify the window to be applied at each frame
    :returns: an array of frames.
    """

    # evaluate the lengths of a frame and of the stride (the number of samples)
    frame_length = int(round(frame_duration * sample_rate))
    frame_stride = int(round(frame_step * sample_rate))

    # evaluate the length (number of samples) of the padding to be added at the end
    padding_length = ((len(signal) // frame_stride) * frame_stride) + frame_length - len(signal)
    padding_length = padding_length if padding_length < frame_length else padding_length - frame_stride

    # evaluate the total number of frames
    number_of_frames = (len(signal) - frame_length) // frame_stride

    # add the zero padding at the signal
    signal = np.append(signal, np.zeros(padding_length))

    if DEBUG:
        print("signal length (padded) = ", len(signal), " samples")
        print("frame length = ", frame_length, " samples")
        print("stride length = ", frame_stride, " samples")
        print("padding length = ", padding_length, " samples")
        print("number of frames = ", number_of_frames)

    frames_no_window = [signal[frame_stride * i:frame_stride * i + frame_length] for i in range(number_of_frames)]

    frames_window = [np.multiply(frames_no_window[i], window_function(frame_length)) for i in range(number_of_frames)]

    if DEBUG:
        index = random.randint(0,number_of_frames)

        plt.figure(1)
        plt.title('frame ' + str(index) + " with no window")
        plt.plot(frames_no_window[index])
        plt.show()

        plt.figure(2)
        plt.title('frame ' + str(index) + " with window")
        plt.plot(frames_window[index])
        plt.show()

    return frames_window


def get_fft_length(frame_length):

    """Evaluate the length of fft to be used, i.e the smallest power of 2 bigger then the frame size
    :param frame_length: the length of a frame (number of samples).
    :returns: the length of the fft to be used .
    """

    fft_size = 1
    while fft_size < frame_length:
        fft_size *= 2

    return fft_size


def power_spectrum(frames, fft_length):
    """For each frame it computes the power spectrum.
    :param frames: the array of frames.
    :param fft_length: the length of fft to be used.
    :returns: An array of the same size of frames with in any cell an array containing the fft of the corresponding
    frame. Just (N/2+1)samples are kept due to hermitian symmetry (where N is the length of fft)
    """
    return np.square(np.absolute(np.fft.rfft(frames, fft_length)))/fft_length


def hertz2mel(hertz):

    """Converts the value of hertz from Hertz to Mels.
    :param hertz: a value or numpy array in Hertz.
    :returns: a value or numpy array in Mels.
    """

    return 1125 * np.log1p(hertz/700)


def mel2hertz(mels):

    """Converts the value of mels from Mels to Hertz.
    :param mels: a value or numpy array in Mels.
    :returns: a value or numpy array in Hertz.
    """

    return 700*(np.expm1(mels/1125))


def get_filter_banks(hertz_from, hertz_to, number_of_filters, fft_length, sample_rate):

    """Returns a Mel filterbank into an array.
    :param hertz_from: the lower frequency to be considered.
    :param hertz_to: the highest frequency to be considered.
    :param number_of_filters: the number of filters in the filterbank.
    :param fft_length: the length of the fft used.
    :param sample_rate: the sample rate of original signal.
    :returns: the filterbank into an array
    """

    mels_from = hertz2mel(hertz_from)
    mels_to = hertz2mel(hertz_to)

    points_mel = np.linspace(mels_from, mels_to, number_of_filters+2)

    points_hz = mel2hertz(points_mel)

    # we need to round these frequency points to the nearest FFT bin because we do not have sufficient frequency
    # resolution

    fft_bins = np.floor((fft_length+1)*points_hz/sample_rate)

    # see http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn2

    filters = np.zeros([number_of_filters, fft_length//2+1])

    for i in range(0, number_of_filters):
        for j in range(int(fft_bins[i]), int(fft_bins[i+1])):
            filters[i, j] = (j-fft_bins[i]) / (fft_bins[i+1]-fft_bins[i])
        for j in range(int(fft_bins[i+1]), int(fft_bins[i+2])):
            filters[i, j] = (fft_bins[i+2]-j) / (fft_bins[i+2]-fft_bins[i+1])

    if DEBUG:
        plt.figure(1)
        plt.title("filters")
        for i in range(0, number_of_filters):
            plt.plot(filters[i])

        plt.show()

    return filters


def get_features(signal,  sample_rate, hertz_from=300, hertz_to = None, number_of_filters=26, pre_emphasis_coef=0.95, frame_duration=0.025, frame_step=0.010,
             window_function=lambda z: np.ones((z,)), power_of_2 = True, useDCT = False):
    """Get MFCC coefficients.
        :param signal: the array to be framed.
        :param pre_emphasis_coef: coefficient for the pre emphasis
        :param hertz_from: the lower frequency to be considered.
        :param hertz_to: the highest frequency to be considered.
        :param number_of_filters: the number of filters in the filterbank.
        :param sample_rate: the sample rate of the original audio signal.
        :param frame_duration: the duration (in seconds) of each frame.
        :param frame_step: the time distance between the beginning of two adjacent frames.
        :param window_function: a function that specify the window to be applied at each frame
        :param power__of_2: True if fft length chosen has the smallest power of two bigger than the frame size
        False if the fft length chosen as the frame size
        :param useDCT: true to get MFCC with the cosine tranform.
        :returns: an array of frames.
        """

    # pre emphasis
    signal = pre_emphasis(signal,pre_emphasis_coef)

    # devide signal into frames
    frames = frame_signal(signal, sample_rate, frame_duration, frame_step, window_function)

    # evaluate fft length
    fft_length = len(frames[0])

    if power_of_2:
        fft_length = get_fft_length(len(frames[0]))

    # evaluate power spectrum for every frame
    power_sp = power_spectrum(frames, fft_length)

    #Mel filterbank

    if hertz_to is None:
        hertz_to = sample_rate/2

    filters = get_filter_banks(hertz_from, hertz_to, number_of_filters, fft_length, sample_rate)

    filterbank_energies = np.dot(power_sp, filters.T)
    #energy = np.sum(power_sp, 1) #frames total energy

    # if some energies are 0 they are replaced with eps because we need to take the log
    filterbank_energies = np.where(filterbank_energies == 0, np.finfo(float).eps, filterbank_energies)
    #energy = np.where(energy == 0, np.finfo(float).eps, energy)

    filterbank_energies = np.log(filterbank_energies)
    #energy = np.log(energy)

    # MFCC if dct must be used
    if useDCT:
        filterbank_energies = dct(filterbank_energies, type=2, axis=1, norm='ortho') # [:, 1:13]

    # delta (velocity coefficients)
    delta = get_delta(filterbank_energies)

    # delta-delta (acceleration coefficients)
    delta_delta = get_delta(delta)

    return np.array([filterbank_energies,delta,delta_delta])


# see http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# for delta formula

def get_delta(features, M = 2):
    """From a matrix of MFCC coefficients compute delta coefficients, from a matrix of delta coefficients
     it computes the delta-delta coefficients.
    :param features: matrix of features (features coefficients for every frame)
    :param M: maximum frame distance to compute the "derivative"
    :returns: matrix with delta coefficients ot with delta-delta coefficients based on input
    """

    # matrix of delta coefficients
    delta = np.zeros(features.shape)

    # pad features to enable the computation of "derivative" in the borders
    features_padded = np.pad(features, ((M, M), (0, 0)), mode='edge')

    d = M*(M+1)*(2*M+1)/3  # the denominator is 2*(1^2 + 2^2 + 3^2 + ... + M^2) but the sum of squares
    # is equal to M*(M+1)*(2*M+1)/6

    for t in range(len(features)):
        for i in range(1, M+1):
            delta[t] = delta[t] + i*(features_padded[t+M+i] - features_padded[t+M-i])

    return delta/d

