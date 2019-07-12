from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os
import util as u
import numpy as np

batch_size = 64
num_classes = 6 #4
epochs = 20
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'Test Model_1.h5'
input_path = "data/"

# The data, split between train and test sets:
# tr, va, te, tr_l, va_l, te_l = u.create_dataset_and_split(input_path,  50, 0.7, (97, 40, 3), max_classes=4)

# tr, va, te, tr_l, va_l, te_l = u.create_dataset_and_split(input_path, n_samples_test=1, training_percentage = 0.8,
#                                                           sample_shape=(64, 64, 3), number_of_filters=64, addDelta=True,
#                                                           frame_duration=0.03, frame_step=0.015 , max_classes = 6,
#                                                           printInfo=True)

tr, va, te, tr_l, va_l, te_l = u.create_dataset_and_split(input_path, n_samples_test=1, training_percentage = 0.8,
                                                          sample_shape=(97, 40), number_of_filters=40, addDelta=False,
                                                          frame_duration=0.025, frame_step=0.010 , max_classes = 6,
                                                          printInfo=True, normalize=False)

te = va
te_l = va_l
tr = np.expand_dims(tr, axis=-1)
te = np.expand_dims(te, axis=-1)

print('x_train shape:', tr.shape)
print(tr.shape[0], 'train samples')
print(te.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
tr_l = keras.utils.to_categorical(tr_l, num_classes)
te_l = keras.utils.to_categorical(te_l, num_classes)

model = Sequential()

#6 classes no delta 64x64 -> train=??? validation = ???
#6 classes delta 64x64 -> train=??? validation = ???
#6 classes no delta 97x40 -> train=??? validation = 93.41%
#6 classes delta 97x40 -> train=??? validation = ???
#6 classes delrta 97x40 normalized -> train=??? validation = ???
#6 classes no delta 97x40 normalized ->
#
model.add(Conv2D(64, (20, 8), padding='same', input_shape=tr.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Conv2D(64, (10, 4), padding='same'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('linear'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#6 classes no delta 64x64 -> train=97.30% validation = 93.67%
#6 classes delta 64x64 -> train=98.35% validation = 93.37%
#6 classes no delta 97x40 -> train=96.83% validation = 93.41%
#6 classes delta 97x40 -> train=97.69% validation = 93.69%
#6 classes delrta 97x40 normalized -> train=96.23% validation = 93.50%
#6 classes no delta 97x40 normalized -> tf vm2
#
# model.add(Conv2D(64, (20, 8), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 3)))
# model.add(Conv2D(64, (10, 4), padding='valid'))
# model.add(Activation('relu'))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))



##### DAVIDE4 all classes (delta) 97x40 -> train 92.56, test = 81.01
##### 64 filters (20,8)
##### relu
##### max pool (1,3)

##### 64 filters (10,4)
##### relu
#####
##### flatten
##### 32 dense linear
##### 128 dense relu
##### softmax



#############   6 classes (no delta) 95% train, 94.38% test
# model.add(Conv2D(64, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3), padding='valid'))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3), padding='valid'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# 2 classes 99.77% train, 99.37% test.... 6 classes train = 97.89% validation = 93.93%
# model.add(Conv2D(128, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='valid', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(512, (3, 3), padding='valid'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# 98.96% training, 100% test....... 2 classes
# model.add(Conv2D(128, (3, 3), padding='valid', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='valid', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(16, (5, 5), padding='valid', strides=(2,2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# 2 class 98% and 100% test (no delta)
# model.add(Conv2D(128, (3, 3), padding='valid', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='valid', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(16, (5, 5), padding='valid', strides=(2,2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# all 64 no stride 2 class = 99.37%, local = 98.12% (no delta)
#
# all classes 20 epochs: train = 78%, test = 82%
#
# model.add(Conv2D(64, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(16, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

#17 epochs 95%
# model = Sequential()
# model.add(BatchNormalization(axis=-1, input_shape=tr.shape[1:]))
# model.add(Conv2D(64, (7, 3), strides=(1, 1), name='conv0'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (7, 3), strides=(1, 1), name='conv0b'))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((1, 3), name='maxpool'))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(32, (3, 3), strides=(1, 1), name='conv1'))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), strides=(1, 1), name='conv1b'))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(16, (3, 3), strides=(1, 1), name='conv2'))
# model.add(Activation('relu'))
# model.add(Conv2D(16, (3, 3), strides=(1, 1), name='conv2b'))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(32, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros', name='linear'))
# model.add(Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros', name='relu'))
# model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros', name='softmax'))

#  14 epochs 93%
# model = Sequential()
# model.add(BatchNormalization(axis=-1, input_shape=tr.shape[1:]))
# model.add(Conv2D(64, (7, 3), strides=(1, 1), name='conv0'))
# model.add(Conv2D(64, (7, 3), strides=(1, 1), name='conv0b'))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((1, 3), name='maxpool'))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(32, (5, 3), strides=(1, 1), name='conv1'))
# model.add(Conv2D(32, (5, 3), strides=(1, 1), name='conv1b'))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(Conv2D(16, (5, 3), strides=(1, 1), name='conv2'))
# model.add(Conv2D(16, (5, 3), strides=(1, 1), name='conv2b'))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(32, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros', name='linear'))
# model.add(Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros', name='relu'))
# model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros', name='softmax'))

#tensorflow 91.5%
# model.add(Conv2D(64, (5, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (5, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 5)))
# model.add(Dropout(0.25))
#
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))


# accuracy 0.955
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 3)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))


##################### 0.92 4 classes 20 epochs ##########
# model.add(Conv2D(32, (5, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 3)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

######Test accuracy: 0.955 (4 classes, 20 epochs) #######################################################
# model.add(Conv2D(32, (5, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (5, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 5)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

######Test accuracy: 0.98 (with all classes test 0.8577777777777778, train 91%) #######################################################
# model.add(Conv2D(32, (7, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (7, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 3)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))


########################Test accuracy 0.95 20 epochs###############################################
#
# model.add(Conv2D(32, (7, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (7, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 3)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

###################### Test accuracy: 0.925 20 epochs##########################################
#
# model.add(Conv2D(64, (7, 3), padding='same', input_shape=tr.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(1, 3)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Activation('linear'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tr = tr.astype('float32')
te = te.astype('float32')
# tr /= 255
# te /= 255

if not data_augmentation:

    model.summary()
    print('Not using data augmentation.')
    model.fit(tr, tr_l,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(te, te_l),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(tr)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(tr, tr_l,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(te, te_l),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(te, te_l, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])