# import keras.backend as K
import numpy as np
from model import load
import features as f
from scipy.io import wavfile
import keras.backend as K

model_path = 'models/22-07-2019,15-09-08'
wav_file = 'files/1.wav'

# access wav file
fs, signal = wavfile.read(wav_file)

# load the models
model = load(model_path, custom_objects={"backend": K})


# get features
features = f.get_time_padded_features(signal, sample_rate=fs,
                                      # PADDING
                                      target_frame_number=110,
                                      random_time_shift=True,
                                      smooth=True,
                                      smooth_length=5,

                                      pre_emphasis_coef=0.95,
                                      # FRAMING PARAMETERS
                                      frame_length=400,
                                      frame_step=160,
                                      window_function=np.hamming,

                                      # MEL FILTERS PARAMETERS
                                      hertz_from=300,
                                      hertz_to=None,
                                      number_of_filters=40,

                                      # FFT PARAMETERS
                                      power_of_2=True,

                                      # OUTPUT SETTINGS
                                      dtype='float32',
                                      use_dct=False,
                                      add_delta=False,

                                      # NORMALIZATION
                                      shift_static=0,
                                      scale_static=1,
                                      shift_delta=0,
                                      scale_delta=1,
                                      shift_delta_delta=0,
                                      scale_delta_delta=1)

# adding batch dim
features = np.expand_dims(features, axis=0)
prediction = model.predict(features)

print(prediction)
