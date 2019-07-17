# Functions to extract filterbank features.
# Authors: Andrea Maracani, Davide Talon. 2019
import numpy as np
import matplotlib.pyplot as plt
import random
import math
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


def get_feature_shape(signal_length, frame_length=None, padding=None, frame_step=None, num_frames=None,
                      number_of_filters=26, add_delta=True):
    """Use known parameters to estimate the shape of a feature image: at least one parameter among frame_length and
    padding must be set and at least one parameter among frame_step and num_frames must be set.

    :param signal_length: the length of the signal.
    :param frame_length: number of samples of each frame.
    :param padding: the number of zeros to be added at the end.
    :param frame_step: number of samples between two adjacent frames
    :param num_frames: total number of frames
    :param number_of_filters: the number of filters in the filterbank.
    :param add_delta: true to include two additional channels, the deltas and the delta-deltas coefficients.
    :returns: the shape of the feature image.
    """

    frame_length, padding, frame_step, num_frames = get_frame_params(signal_length, frame_length, padding, frame_step,
                                                                     num_frames)

    if add_delta:
        return num_frames, number_of_filters, 3

    return num_frames, number_of_filters, 1


def get_frame_params(signal_length, frame_length=None, padding=None, frame_step=None, num_frames=None):
    """Use known parameters to estimate others for framing: at least one parameter among frame_length and padding must
    be set and at least one parameter among frame_step and num_frames must be set.

    :param signal_length: the length of the signal to be framed.
    :param frame_length: number of samples of each frame.
    :param padding: the number of zeros to be added at the end.
    :param frame_step: number of samples between two adjacent frames
    :param num_frames: total number of frames
    :returns: the framing parameters in the same order as arguments.
    """

    # check that parameters are correct
    assert signal_length > 0, "Signal length must be greater then 0!"
    assert (frame_length is not None) or (padding is not None), \
        "At least one of frame_length and padding must be set!"
    assert (frame_step is not None) or (num_frames is not None), \
        "At least one of frame_step and num_frames must be set!"

    # when frame length specified we limit the padding to this maximum of percentage of a frame
    percentage = 0.5

    if frame_step is None and frame_length is not None:
        frame_step = int(math.ceil((signal_length - percentage * frame_length) / (num_frames - 1))) - 1
    elif frame_step is None:
        frame_step = int(math.ceil(signal_length / (num_frames - 1))) - 1
    elif num_frames is None and frame_length is not None:
        # num_frames = int(math.ceil(1 - frame_length*percentage + signal_length / frame_step)) - 1
        num_frames = int(round(1 + (signal_length - frame_length * percentage) / frame_step))
    elif num_frames is None:
        num_frames = int(math.ceil(1 + signal_length / frame_step)) - 1

    if frame_length is None:
        frame_length = signal_length + padding - (num_frames - 1) * frame_step
    elif padding is None:
        padding = (num_frames - 1) * frame_step + frame_length - signal_length

    return frame_length, padding, frame_step, num_frames


def frame_signal(signal, frame_length=None, padding=None, frame_step=None, num_frames=None,
                 window_function=lambda z: np.ones((z,))):
    """Divide a signal into frames: at least one parameter among frame_length and padding must be set and at least one
    parameter among frame_step and num_frames must be set.

    :param signal: the array to be framed.
    :param frame_length: number of samples of each frame.
    :param padding: number of samples to be padded at the end.
    :param frame_step: number of samples between two adjacent frames
    :param num_frames: total number of frames
    :param window_function: a function that specify the window to be applied at each frame
    :returns: an array of frames.
    """

    frame_length, padding, frame_step, num_frames = get_frame_params(len(signal), frame_length, padding, frame_step,
                                                                     num_frames)

    if DEBUG:
        print("**************FRAMING DEBUG***************************")
        print("signal length = ", len(signal), " samples")
        print("frame length = ", frame_length, " samples")
        print("stride length = ", frame_step, " samples")
        print("padding length = ", padding, " samples")
        print("number of frames = ", num_frames)
        print("******************************************************")

    # add the zero padding at the signal
    signal = np.append(signal, np.zeros(padding))

    frames_no_window = [signal[frame_step * i:frame_step * i + frame_length] for i in range(num_frames)]

    frames_window = [np.multiply(frames_no_window[i], window_function(frame_length)) for i in range(num_frames)]

    if DEBUG:
        index = random.randint(0, num_frames)

        plt.figure(1)
        plt.title('frame ' + str(index) + " with no window")
        plt.plot(frames_no_window[index])
        plt.show()

        plt.figure(2)
        plt.title('frame ' + str(index) + " with window")
        plt.plot(frames_window[index])
        plt.show()

    return frames_window


##################OLD VERSION ######################################
# def frame_signal(signal, sample_rate, frame_duration=0.025, frame_step=0.010, window_function=lambda z: np.ones((z,))):
#     """Divide a signal into frames.
#     :param signal: the array to be framed.
#     :param sample_rate: the sample rate of the original audio signal.
#     :param frame_duration: the duration (in seconds) of each frame.
#     :param frame_step: the time distance between the beginning of two adjacent frames.
#     :param window_function: a function that specify the window to be applied at each frame
#     :returns: an array of frames.
#     """
#
#     # evaluate the lengths of a frame and of the stride (the number of samples)
#     frame_length = int(round(frame_duration * sample_rate))
#     frame_stride = int(round(frame_step * sample_rate))
#
#     # evaluate the length (number of samples) of the padding to be added at the end
#     padding_length = ((len(signal) // frame_stride) * frame_stride) + frame_length - len(signal)
#     padding_length = padding_length if padding_length < frame_length else padding_length - frame_stride
#
#     # evaluate the total number of frames
#     number_of_frames = (len(signal) - frame_length) // frame_stride
#
#     # add the zero padding at the signal
#     signal = np.append(signal, np.zeros(padding_length))
#
#     if DEBUG:
#         print("signal length (padded) = ", len(signal), " samples")
#         print("frame length = ", frame_length, " samples")
#         print("stride length = ", frame_stride, " samples")
#         print("padding length = ", padding_length, " samples")
#         print("number of frames = ", number_of_frames)
#
#     frames_no_window = [signal[frame_stride * i:frame_stride * i + frame_length] for i in range(number_of_frames)]
#
#     frames_window = [np.multiply(frames_no_window[i], window_function(frame_length)) for i in range(number_of_frames)]
#
#     if DEBUG:
#         index = random.randint(0,number_of_frames)
#
#         plt.figure(1)
#         plt.title('frame ' + str(index) + " with no window")
#         plt.plot(frames_no_window[index])
#         plt.show()
#
#         plt.figure(2)
#         plt.title('frame ' + str(index) + " with window")
#         plt.plot(frames_window[index])
#         plt.show()
#
#     return frames_window


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
    return np.square(np.absolute(np.fft.rfft(frames, fft_length))) / fft_length


def hertz2mel(hertz):
    """Converts the value of hertz from Hertz to Mels.
    :param hertz: a value or numpy array in Hertz.
    :returns: a value or numpy array in Mels.
    """

    return 1125 * np.log1p(hertz / 700)


def mel2hertz(mels):
    """Converts the value of mels from Mels to Hertz.
    :param mels: a value or numpy array in Mels.
    :returns: a value or numpy array in Hertz.
    """

    return 700 * (np.expm1(mels / 1125))


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

    points_mel = np.linspace(mels_from, mels_to, number_of_filters + 2)

    points_hz = mel2hertz(points_mel)

    # we need to round these frequency points to the nearest FFT bin because we do not have sufficient frequency
    # resolution

    fft_bins = np.floor((fft_length + 1) * points_hz / sample_rate)

    # see http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn2

    filters = np.zeros([number_of_filters, fft_length // 2 + 1])

    for i in range(0, number_of_filters):
        for j in range(int(fft_bins[i]), int(fft_bins[i + 1])):
            filters[i, j] = (j - fft_bins[i]) / (fft_bins[i + 1] - fft_bins[i])
        for j in range(int(fft_bins[i + 1]), int(fft_bins[i + 2])):
            filters[i, j] = (fft_bins[i + 2] - j) / (fft_bins[i + 2] - fft_bins[i + 1])

    if DEBUG:
        plt.figure(1)
        plt.title("filters")
        for i in range(0, number_of_filters):
            plt.plot(filters[i])

        plt.show()

    return filters


def get_features_filters(signal, filters, pre_emphasis_coef=0.95,
                         # FRAMING PARAMETERS
                         frame_length=None, padding=None, frame_step=None, num_frames=None, window_function=np.hamming,
                         # FFT PARAMETERS
                         power_of_2=True,
                         # OUTPUT SETTINGS
                         dtype='float32', use_dct=False, add_delta=True,

                         #NORMALIZATION PARAMETERS
                         shift_static=0, scale_static=1.0,
                         shift_delta=0, scale_delta=1.0,
                         shift_delta_delta=0, scale_delta_delta=1.0):
    """Get MFCC coefficients.
        :param signal: the array to be framed.
        :param filters: the Mel filterbanks.
        :param pre_emphasis_coef: coefficient for the pre emphasis

        :param frame_length: number of samples of each frame (at least one among this and padding must be set).
        :param padding: number of samples to be padded at the end (at least one among this and frame_length must be set).
        :param frame_step: number of samples between two adjacent frames (at least one among this and num_frames must
        be set).
        :param num_frames: total number of frames (at least one among this and frame_step must be set).
        :param window_function: a function that specifies the window to be applied at each frame

        :param power_of_2: True if fft length chosen has the smallest power of two bigger than the frame size
        False if the fft length chosen as the frame size

        :param dtype: the type of elements of output... use float32 or float64
        :param use_dct: true to get MFCC with the cosine transform.
        :param add_delta: true to include two additional channels, the deltas and the delta-deltas coefficients.

        :param shift_static: shift the static feature of this amount.
        :param scale_static: scale the static feature of this amount.
        :param shift_delta: shift the delta feature of this amount.
        :param scale_delta: scale the delta feature of this amount.
        :param shift_delta_delta: shift the delta delta feature of this amount.
        :param scale_delta_delta: scale the delta delta feature of this amount.

        :returns: an array of frames.
        """

    # pre emphasis
    signal = pre_emphasis(signal, pre_emphasis_coef)

    # divide signal into frames
    frames = frame_signal(signal, frame_length, padding, frame_step, num_frames, window_function)

    # evaluate fft length
    fft_length = len(frames[0])

    if power_of_2:
        fft_length = get_fft_length(len(frames[0]))

    # evaluate power spectrum for every frame
    power_sp = power_spectrum(frames, fft_length)

    # use filters to get energies
    filterbank_energies = np.dot(power_sp, filters.T)

    # if some energies are 0 they are replaced with eps because we need to take the log
    filterbank_energies = np.where(filterbank_energies == 0, np.finfo(float).eps, filterbank_energies)

    filterbank_energies = np.log(filterbank_energies)

    # if we should use DCT
    if use_dct:
        filterbank_energies = dct(filterbank_energies, type=2, axis=1, norm='ortho')

    # if delta and delta-delta should be added
    if add_delta:
        # delta (velocity coefficients)
        delta = (get_delta(filterbank_energies) + shift_delta)*scale_delta

        # delta-delta (acceleration coefficients)
        delta_delta = (get_delta(delta) + shift_delta_delta)*scale_delta_delta

        filterbank_energies = (filterbank_energies + shift_static)*scale_static

        features = np.array([filterbank_energies, delta, delta_delta], dtype=dtype)
        return np.transpose(features, (1, 2, 0))

    return np.transpose(np.array([(filterbank_energies + shift_static)*scale_static], dtype=dtype), (1, 2, 0))


def get_time_padded_features(signal, sample_rate,
                             # PADDING
                             target_frame_number, random_time_shift=True, smooth=True, smooth_length=10,

                             pre_emphasis_coef=0.95,
                             # FRAMING PARAMETERS
                             frame_length=None, frame_step=None, window_function=np.hamming,
                             # MEL FILTERS PARAMETERS
                             hertz_from=300, hertz_to=None, number_of_filters=26,
                             # FFT PARAMETERS
                             power_of_2=True,
                             # OUTPUT SETTINGS
                             dtype='float32', use_dct=False, add_delta=True,
                             # NORMALIZATION
                             shift_static=0, scale_static=1.0,
                             shift_delta=0, scale_delta=1.0,
                             shift_delta_delta=0, scale_delta_delta=1.0):

    # get features not padded
    features = get_features(signal, sample_rate, pre_emphasis_coef=pre_emphasis_coef, frame_length=frame_length,
                            padding=None, frame_step=frame_step, num_frames=None, window_function=window_function,
                            hertz_from=hertz_from, hertz_to=hertz_to, number_of_filters=number_of_filters,
                            power_of_2=power_of_2, dtype=dtype, use_dct=use_dct, add_delta=add_delta,
                            shift_static=shift_static, scale_static=scale_static, shift_delta=shift_delta,
                            scale_delta=scale_delta, shift_delta_delta=shift_delta_delta,
                            scale_delta_delta=scale_delta_delta)

    # the actual number of frames
    number_of_frames = features.shape[0]

    # the number of channels
    channels = features.shape[2]

    # the size of the padding needed
    pad_size = target_frame_number - number_of_frames

    # initialize output
    features_padded = np.empty((target_frame_number, number_of_filters, channels), dtype=dtype)

    # check the pad size is positive
    assert pad_size >= 0, "pad size negative!"

    # initialize left and right pad
    pad_left = int(math.ceil(pad_size/2))
    pad_right = pad_size - pad_left

    # if random padding
    if random_time_shift:
        pad_left = np.random.randint(0, pad_size)
        pad_right = pad_size - pad_left

    for i in range(channels):

        # pad with minimum static feature and with 0 delta and delta delta features
        if i == 0:
            pad_value = np.min(features[:, :, i])
        else:
            pad_value = 0

        # smooth the edges
        if smooth:

            # left smooth
            smooth_in = np.array([i/smooth_length for i in range(smooth_length)])
            smooth_in = np.transpose(np.tile(smooth_in, (number_of_filters,1)))
            features[0:smooth_length, :, i] = np.multiply(features[0:smooth_length, :, i] - pad_value,
                                                          smooth_in) + pad_value
            # right smooth
            smooth_out = np.flip(smooth_in, 0)
            features[-smooth_length:, :, i] = np.multiply(features[-smooth_length:, :, i] - pad_value,
                                                          smooth_out) + pad_value

        # add padding
        features_padded[:, :, i] = np.pad(features[:, :, i], [(pad_left, pad_right), (0, 0)], mode="constant",
                                          constant_values=[(pad_value, pad_value)])

    return features_padded


def get_features(signal, sample_rate, pre_emphasis_coef=0.95,  # SIGNAL AND PRE EMPHASIS
                 # FRAMING PARAMETERS
                 frame_length=None, padding=None, frame_step=None, num_frames=None, window_function=np.hamming,
                 # MEL FILTERS PARAMETERS
                 hertz_from=300, hertz_to=None, number_of_filters=26,
                 # FFT PARAMETERS
                 power_of_2=True,
                 # OUTPUT SETTINGS
                 dtype='float32', use_dct=False, add_delta=True,

                 # NORMALIZATION PARAMETERS
                 shift_static = 0, scale_static = 1.0,
                 shift_delta = 0, scale_delta = 1.0,
                 shift_delta_delta = 0, scale_delta_delta = 1.0):

    """Get MFCC coefficients.
        :param signal: the array to be framed.
        :param sample_rate: the sample rate of the signal.
        :param pre_emphasis_coef: coefficient for the pre emphasis

        :param frame_length: number of samples of each frame (at least one among this and padding must be set).
        :param padding: number of samples to be padded at the end (at least one among this and frame_length must be set).
        :param frame_step: number of samples between two adjacent frames (at least one among this and num_frames must
        be set).
        :param num_frames: total number of frames (at least one among this and frame_step must be set).
        :param window_function: a function that specifies the window to be applied at each frame

        :param hertz_from: the lower frequency to be considered.
        :param hertz_to: the highest frequency to be considered.
        :param number_of_filters: the number of filters in the filterbank.

        :param power_of_2: True if fft length chosen has the smallest power of two bigger than the frame size
        False if the fft length chosen as the frame size

        :param dtype: the type of elements of output... use float32 or float64
        :param use_dct: true to get MFCC with the cosine tranform.
        :param add_delta: true to include two additional channels, the deltas and the delta-deltas coefficients.

        :param shift_static: shift the static feature of this amount.
        :param scale_static: scale the static feature of this amount.
        :param shift_delta: shift the delta feature of this amount.
        :param scale_delta: scale the delta feature of this amount.
        :param shift_delta_delta: shift the delta delta feature of this amount.
        :param scale_delta_delta: scale the delta delta feature of this amount.

        :returns: an array of frames.
        """

    # complete parameters
    frame_length_, padding_, frame_step_, num_frames_ = get_frame_params(len(signal), frame_length, padding, frame_step,
                                                                         num_frames)
    # evaluate fft length
    fft_length = frame_length_

    if power_of_2:
        fft_length = get_fft_length(frame_length_)

    if hertz_to is None:
        hertz_to = sample_rate / 2

    filters = get_filter_banks(hertz_from, hertz_to, number_of_filters, fft_length, sample_rate)

    return get_features_filters(signal, filters, pre_emphasis_coef, frame_length, padding, frame_step, num_frames,
                                window_function, power_of_2, dtype, use_dct, add_delta, shift_static, scale_static,
                                shift_delta, scale_delta, shift_delta_delta, scale_delta_delta)


# see http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# for delta formula

def get_delta(features, M=2):
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

    d = M * (M + 1) * (2 * M + 1) / 3  # the denominator is 2*(1^2 + 2^2 + 3^2 + ... + M^2) but the sum of squares
    # is equal to M*(M+1)*(2*M+1)/6

    for t in range(len(features)):
        for i in range(1, M + 1):
            delta[t] = delta[t] + i * (features_padded[t + M + i] - features_padded[t + M - i])

    return delta / d
