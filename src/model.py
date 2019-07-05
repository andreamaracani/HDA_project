import numpy as np
import tensorflow as tf
import os
import datetime
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint

class ASRModel(object):

    def __init__(self, architecture, input_size, **params):

        """
        Initialize the model with the desired architecture

        Args:
            architecture: architecture required
            input_size: size of the input to the model

        Raises:
            Exception: If the architecture type isn't recognized.
        """
        
        if (architecture == 'toy-model'):
            self.architecture = toy_model(input_size, **params)
        elif (architecture == 'cnn-trad-fpool3'):
            self.architecture = cnn_trad_fpool3(input_size, **params)
        else:
            raise Exception('Model architecture not recognized')
        
    def save(self):

        """
        Save the model into models/YYYY-MM-DD HH/mm/ss

        Args:
            model: model to save
        """ 
        if 'models' not in os.listdir():
            os.mkdir('models')
        
        date = datetime.datetime.now()
        self.architecture.save('models/' + str(date))
        return date


def load(filepath):
    """
    Load the model from the filepath

    Args:
        model: model to save
    """ 
    model = load_model(filepath)

    return model


def toy_model(input_shape, **params):

    """
    Create a CNN model:
    [padding] -> [Conv] -> [Batch-norm] -> [Relu] -> [MaxPool] -> [softmax]


    Args:
        input_shape: tuple of ints indicating the shape of the input
    
    Returns:
        an instance of the model created
    """

    # placeholder for the input
    X_input = Input(input_shape)

    # padding the input
    X = ZeroPadding2D((3, 3))(X_input)

    # Conv0 layer
    X = Conv2D(32, (7,7), strides=(1,1), name='conv0')(X)
    X = BatchNormalization(axis=-1, name='bn0')(X)
    X = Activation('relu')(X)

    # maxpooling
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool')(X)

    # flattening x
    X = Flatten()(X)

    # Fully connected
    X = Dense(5, activation='softmax', name='softmax-layer')(X)

    # return the model instance.
    model = Model(inputs = X_input, outputs = X, name='1-conv-model')

    return model


def cnn_trad_fpool3(input_shape, **params):

    """
    Create the cnn-trad-fpool3 model as Convolutional Neural Networks for Small-fooprint Keyword Spotting
    [Sainath15]:

    [Conv] -> [MaxPool] -> [Conv] -> [Linear] -> [Relu] -> [softmax]
    

    Args:
        input_shape: tuple of ints indicating the shape of the input
    
    Returns:
        an instance of the model created
    """
    # input shape: (batch_size, time, freq, channels)
    X_input = Input(input_shape)

    # conv0:convolution layer with 64 filters, kernel size freq=64, time=9, stride(1, 1)
    X = Conv2D(64, (64, 8), strides=(1,1), name='conv0')(X_input)

    # pooling in frequency within a region of t=1, f=3
    X = MaxPooling2D((1, 3), name='maxpool')(X)

    # conv1:convolution layer with 64 filters, kernel size freq=32, time=4, stride(1, 1)
    X = Conv2D(64, (32, 4), strides=(1, 1), name='conv1')(X)
    
    # flatten the filters
    X = Flatten()(X)

    # linear: linear layer with 32 units
    X = Dense(32, activation='linear', name='linear')(X)

    # relu: fully connected layer with 128 relu activation units
    X = Dense(128, activation='relu', name='relu')(X)

    # softmax: softmax layer
    X = Dense(4, activation='softmax', name='softmax')(X)

    model = Model(inputs = X_input, outputs = X, name='1-conv-model')

    # return the model instance
    return model


# test of functionalities
if __name__ == "__main__":

    # input shapes
    input_size = 12
    frames = 97
    m = 1000
    m_test= 150

    # creating random tensors for train and test
    X_train = np.random.randn(m, frames, input_size, 3)
    Y_train = np.random.randint(0, 4, m)

    X_test = np.random.randn(m_test, frames, input_size, 3)
    Y_test = np.random.randint(0, 4, m_test)

    params = 5
   
    # initialize the computational graph
    model = ASRModel(architecture='toy-model', input_size=(frames, input_size, 3), params=params)
    cnn_model = model.architecture

    cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # training
    checkpoint = ModelCheckpoint('models/checkpoints', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=3)
    cnn_model.fit(x = X_train, y = Y_train, epochs=7, batch_size=3, callbacks=[checkpoint])

    # evaluate the model with test set
    preds = cnn_model.evaluate(X_test, Y_test)

    date = model.save()
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    cnn_model = load('models/' + str(date))
    
    # print the summary of the layers(parameters)
    cnn_model.summary()