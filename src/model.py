import numpy as np
import tensorflow as tf
import os
import datetime
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, Concatenate
from keras.layers import Permute, LSTM, Reshape, RepeatVector, Lambda, Multiply, GRU, Conv1D, concatenate, TimeDistributed, ZeroPadding1D, LocallyConnected1D, ZeroPadding2D
from keras.models import Model, Sequential
from keras.models import save_model, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

class ASRModel(object):

    def __init__(self, architecture, input_size, out_size,  **params):

        """
        Initialize the model with the desired architecture

        Args:
            architecture: architecture required
            input_size: size of the input to the model

        Raises:
            Exception: If the architecture type isn't recognized.
        """
        
        if (architecture == 'toy-model'):
            self.architecture = toy_model(input_size, out_size, **params)
        elif (architecture == 'cnn-trad-fpool3'):
            self.architecture = cnn_trad_fpool3(input_size, out_size, **params)
        elif (architecture == 'module-network'):
            self.architecture = module_model(input_size, out_size, **params)
        elif (architecture == 'improved-cnn-trad-fpool3'):
            self.architecture = improved_cnn_trad_fpool3(input_size, out_size, **params)
        elif (architecture == 'sequential-model'):
            self.architecture = sequential_model(input_size, out_size, **params)
        elif (architecture == 'rnn'):
            self.architecture = rnn(input_size, out_size, **params)
        elif (architecture == 'cnn-attention-rnn'):
            self.architecture = cnn_attention_rnn(input_size, out_size, **params)
        elif (architecture == 'inception'):
            self.architecture = inception(input_size, out_size, **params)
        elif (architecture == 'svdf'):
            self.architecture = svdf(input_size, out_size, **params)
        else:
            raise Exception('Model architecture not recognized')
        
        sequential_model
    def save(self):

        """
        Save the model into models/YYYY-MM-DD HH/mm/ss

        Args:
            model: model to save
        """ 
        if 'models' not in os.listdir():
            os.mkdir('models')
        
        date = datetime.datetime.now()
        self.architecture.save('models/' + date.strftime("%d-%m-%Y,%H-%M-%S"))
        return date


def load(filepath):
    """
    Load the model from the filepath

    Args:
        model: model to save
    """ 
    model = load_model(filepath)

    return model


def toy_model(input_shape, out_size, **params):

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

    # flattening x
    X = Flatten()(X)

    # Fully connected
    X = Dense(128, activation='softmax', name='softmax-layer')(X)

    X = Dense(out_size, activation='softmax', name='softmax-layer')(X)


    # return the model instance.
    model = Model(inputs = X_input, outputs = X, name='1-conv-model')

    return model


def cnn_trad_fpool3(input_shape, out_size, **params):

    """
    Create the cnn-trad-fpool3 model as Convolutional Neural Networks for Small-fooprint Keyword Spotting
    [Sainath15]:
    [Conv] -> [MaxPool] -> [Conv] -> [Linear] -> [Relu] -> [softmax]
    
    Args:
        input_shape: tuple of ints indicating the shape of the input
    
    Returns:
        an instance of the model created
    """

    dropout_prob = params['dropout_prob']

    # input shape: (batch_size, time, freq, channels)
    X_input = Input(input_shape)

    # conv0:convolution layer with 64 filters, kernel size freq=64, time=9, stride(1, 1)
    X = Conv2D(64, (20, 8), strides=(1,1), name='conv0')(X_input)

    # non-linearity
    X = Activation('relu')(X)

    # pooling in frequency within a region of t=1, f=3
    X = MaxPooling2D((1, 3), name='maxpool')(X)

    X = Dropout(dropout_prob)(X)
    # conv1:convolution layer with 64 filters, kernel size freq=32, time=4, stride(1, 1)
    X = Conv2D(64, (10, 4), strides=(1, 1), name='conv1')(X)

    # non-linearity
    X = Activation('relu')(X)
    
    # flatten the filters
    X = Flatten()(X)

    # linear: linear layer with 32 units
    X = Dense(32, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros', name='linear')(X)

    # relu: fully connected layer with 128 relu activation units
    X = Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros', name='relu')(X)

    X = Dropout(dropout_prob)(X)

    # softmax: softmax layer
    X = Dense(out_size, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros', name='softmax')(X)

    model = Model(inputs = X_input, outputs = X, name='1-conv-model')

    # return the model instance
    return model


def improved_cnn_trad_fpool3(input_shape, out_size, **params):

    """
    Create an improved version of cnn-trad-fpool3 model from Convolutional Neural Networks for Small-fooprint Keyword Spotting
    [Sainath15]:
    
    Args:
        input_shape: tuple of ints indicating the shape of the input
    
    Returns:
        an instance of the model created
    """

    dropout_prob = params['dropout_prob']

    # input shape: (batch_size, time, freq, channels)
    X_input = Input(input_shape)

    # X = ZeroPadding2D((0, 4))(X_input)

    # normalizing batch
    X = BatchNormalization(axis=-1)(X_input)

    # conv0:convolution layer with 64 filters, kernel size freq=64, time=9, stride(1, 1)
    X = Conv2D(64, (7, 3), strides=(1, 1), name='conv0')(X)
    x = Activation('relu')(X)
    X=  Conv2D(64, (7, 3), strides=(1, 1), name='conv0b')(X)

    # normalizing batch
    X = BatchNormalization(axis=-1)(X)

    # non-linearity
    X = Activation('relu')(X)

    # pooling in frequency within a region of t=1, f=3
    X = MaxPooling2D((1, 3), name='maxpool')(X)

    # Dropout
    X= Dropout(dropout_prob)(X)

    # conv1:convolution layer with 64 filters, kernel size freq=32, time=4, stride(1, 1)
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv1b')(X)

    # normalizing batch
    X = BatchNormalization(axis=-1)(X)

    # non-linearity
    X = Activation('relu')(X)

    X = Dropout(dropout_prob)(X)

    # conv2:convolution layer with 64 filters, kernel size freq=32, time=4, stride(1, 1)
    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv2')(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv2b')(X)

    # normalizing batch
    X = BatchNormalization(axis=-1)(X)

    # non-linearity
    X = Activation('relu')(X)

    X = Dropout(dropout_prob)(X)
    
    # flatten the filters
    X = Flatten()(X)

    # linear: linear layer with 32 units
    X = Dense(32, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros', name='linear')(X)

    # relu: fully connected layer with 128 relu activation units
    X = Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros', name='relu')(X)

    # softmax: softmax layer
    X = Dense(out_size, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros', name='softmax')(X)

    model = Model(inputs = X_input, outputs = X, name='1-conv-model')

    # return the model instance
    return model


def sequential_model(input_shape, out_size, **params):

    dropout_prob = params['dropout_prob']

    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=input_shape))
    model.add(Conv2D(64, (7, 3), strides=(1, 1), name='conv0'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (7, 3), strides=(1, 1), name='conv0b'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 3), name='maxpool'))
    model.add(Dropout(dropout_prob))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), name='conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), name='conv1b'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_prob))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), name='conv2'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), name='conv2b'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_prob))
    model.add(Flatten())
    model.add(Dense(32, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros', name='linear'))
    model.add(Dense(128, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros', name='relu'))
    model.add(Dense(out_size, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros', name='softmax'))

    return model




    """
    Implements a convolutional network as composition of convolutional blocks

    Args:
        input_shape: size of the input to the network
        params: dictionary of parameters that defines the structure

    Returns:
        the convolutional block
    """

    # retrieving the params
    pooling_size = params['pooling_size']
    stride = params['stride']
    kernel = params['kernel']
    filters = params['filters']
    hidden_layers = params['hidden_layers']
    dropout_prob = params['dropout_prob']

    if(len(filters) != hidden_layers):
        raise Exception('The number of filters must be equal to the number of blocks')
    
    print(input_size)
    # placeholder for input
    X_input = Input(input_size)

    # adding the first convolutional layer
    X = block(input_size, filters[0], kernel=(3, 3), strides=stride, pooling_size=pooling_size)(X_input)
    input_size = X.shape[1:]
    # print(X.shape)
    for layer in range(hidden_layers-1):
        print('ciao')
        print(input_size)
        X = block(input_size, filters[layer+1], kernel=kernel, strides=stride, pooling_size=pooling_size, dropout_prob=dropout_prob)(X)
        input_size = X.shape[1:]
    # flatten the filters
    X = Flatten()(X)

    # linear: linear layer with 32 units
    X = Dense(64, activation='linear', name='linear')(X)

    # relu: fully connected layer with 128 relu activation units
    X = Dense(128, activation='relu', name='relu')(X)

    # softmax: softmax layer
    X = Dense(out_size, activation='softmax', name='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    return model


def block(input_size, filters, kernel=(3, 3), strides=(1, 1), pooling_size=(1, 1), dropout_prob=0.3):
    """
    Implements a convolutional block composed as
    [Conv] -> [BatchNorm] -> [MaxPool] -> [ReLU]
    Args:
        input_size: size of the input to the convolutional block
        output_size: size of the output of the convolutional block
        kernel: tuple (h, w) indicating the dimension of the 2d-kernel
        stride: tuple indicating the stride of the convolutional kernel
        pooling_size: tuple indicating the dimension of the pool
        dropout_prob: dropout probability after the convolutional block
    Returns:
        the convolutional block
    """
    # placeholder for the input
    X_input = Input(input_size)
    print(input_size)
    print(X_input.shape)

    X = Conv2D(filters, kernel_size=kernel, padding ='same', strides=strides)(X_input)
    
    X = BatchNormalization(axis=-1)(X)

    X = MaxPooling2D(pooling_size)(X)

    X = Activation('relu')(X)

    # TODO: need to pass the seed
    X = Dropout(1-dropout_prob, noise_shape=None, seed=None)(X)

    model = Model(inputs=X_input, outputs=X)

    return model

def module_model(input_size, out_size, **params):

    """
    Implements a convolutional network as composition of convolutional blocks
    Args:
        input_shape: size of the input to the network
        params: dictionary of parameters that defines the structure
    Returns:
        the convolutional block
    """

    # retrieving the params
    pooling_size = params['pooling_size']
    stride = params['stride']
    kernel = params['kernel']
    filters = params['filters']
    hidden_layers = params['hidden_layers']
    dropout_prob = params['dropout_prob']

    if(len(filters) != hidden_layers):
        raise Exception('The number of filters must be equal to the number of blocks')

    # placeholder for input
    X_input = Input(input_size)
    print(input_size)
    # adding the first convolutional layer
    X = block(input_size, filters[0], kernel=kernel, strides=stride, pooling_size=pooling_size)(X_input)
    input_size = X.shape[1:]
    for layer in range(hidden_layers-1):
        X = block(input_size, filters[layer+1], kernel=kernel, strides=stride, pooling_size=pooling_size, dropout_prob=dropout_prob)(X)
        input_size = X.shape[1:]
    # flatten the filters
    X = Flatten()(X)

    # linear: linear layer with 32 units
    X = Dense(64, activation='linear', name='linear')(X)

    # relu: fully connected layer with 128 relu activation units
    X = Dense(128, activation='relu', name='relu')(X)

    # softmax: softmax layer
    X = Dense(4, activation='softmax', name='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    return model

def rnn(input_size, out_size, **params):

    X_input = Input(input_size)

    # can't feed 4 dim data, let's reshape the input
    X = Reshape((input_size[0], input_size[1]*input_size[2]))(X_input)

    X = GRU(128, return_sequences=True, name='rnn1')(X)
    X = GRU(256, return_sequences=False, name='rnn2')(X)
    X = Dense(out_size, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    return model

def attention_3d_block(input_size, inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    print(a.shape)
    # a = Reshape((input_dim, input_size[0]))(a) # this line is not useful. It's just to know which dimension is what.
    # a = Dense(input_size[0], activation='softmax')(a)

    e = LSTM(64, name='embedding-lstm', return_sequences=True)(a)
    e = Dense(input_size[0], activation='softmax')(e)


    a_probs = Permute((2, 1), name='attention_vec')(e)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def cnn_attention_rnn(input_size, out_size, **params):

    dropout_prob = params['dropout_prob']
    X_input = Input(input_size)

    X = BatchNormalization(axis=-1)(X_input)
    # conv0:convolution layer with 64 filters, kernel size freq=64, time=9, stride(1, 1)
    X = Conv2D(64, (5, 3), strides=(1, 1), padding='same', name='conv0')(X)
    X=  Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv0b')(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((1, 3), name='maxpool0')(X)
    X= Dropout(dropout_prob)(X)

    # conv1:convolution layer with 64 filters, kernel size freq=32, time=4, stride(1, 1)
    X = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv1b')(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((1, 3), name='maxpool1')(X)
    X= Dropout(dropout_prob)(X)


    # attentiont mechanism
    shape = (input_size[0], 4 * 32)
    X = Reshape((shape[0], shape[1]))(X)
    print(X.shape)
    attention_mul = attention_3d_block(shape, X)
    
    # recurrent layer 
    lstm_units = 128
    attention_mul = LSTM(lstm_units, name='rnn', return_sequences=False)(attention_mul)
    output = Dense(out_size, activation='softmax')(attention_mul)
    model = Model(input=X_input, output=output)
    return model

def inception(input_size, out_size, **params):

    X_input = Input(input_size)

    X = Conv2D(16, (1, 1), strides=(1,1), activation='relu', padding='same')(X_input)
    X = Conv2D(16, (1, 3), strides=(1,1), activation='relu', padding='same')(X)
    X = Conv2D(16, (3, 1), strides=(1,1), activation='relu', padding='same')(X)
    X = Conv2D(32, (1, 3), strides=(1,1), activation='relu', padding='same')(X)
    X = Conv2D(32, (3, 1), strides=(1,1), activation='relu', padding='same')(X)

    Y = Conv2D(16, (1, 1), strides=(1,1), activation='relu' )(X_input)
    Y = Conv2D(16, (1, 3), strides=(1,1), activation='relu', padding='same')(Y)
    Y = Conv2D(16, (3, 1), strides=(1,1), activation='relu', padding='same')(Y)

    W = MaxPooling2D((3,3), strides=(1,1), padding='same')(X_input)
    W = Conv2D(16, (1, 1), strides=(1,1), activation='relu', padding='same')(W)

    Z = Conv2D(16, (1, 1), strides=(1,1), activation='relu', padding='same')(X_input)

    output = concatenate([X, Y, W, Z], axis=-1)

    output  =Flatten()(output)
    output = Dense(out_size, activation='softmax')(output)

    model = Model(inputs=X_input, output=output)

    return model
   
def svdf1rank_layer(input_size, num_nodes, memory_size):   

    # input_size = (num_frames, num_coeff, channels)

    X_input = Input(input_size)

    X = Reshape((input_size[0],input_size[1] * input_size[2],1))(X_input)
    X = ZeroPadding2D((memory_size, 0))(X)
    prova = []
    for t in range(input_size[0]):
        Y = Lambda(lambda x: x[:, t: t + memory_size, :])(X)
        Y = Reshape((memory_size * input_size[1] * input_size[2], 1))(Y)
        Y = Conv1D(num_nodes, input_size[1]*input_size[2], strides=input_size[1]*input_size[2])(Y)
        Y = Reshape((memory_size * num_nodes, 1))(Y)
        Y = Conv1D(1, memory_size, strides=memory_size)(Y)
        Y = Reshape((1, num_nodes))(Y)
        prova.append(Y)

    X = Concatenate(axis=1)(prova)
    X = Activation('relu')(X)
    model = Model(inputs=X_input, outputs=X)

    # return the model instance
    return model

def svdf(input_size, out_size, **params):   

    # input_size = (num_frames, num_coeff, channels)

    X_input = Input(input_size)

    X = svdf1rank_layer(input_size, 400, memory_size=8)(X_input)
    X = svdf1rank_layer((input_size[0], 400, 1), 400, memory_size=8)(X)
    X = svdf1rank_layer((input_size[0], 400, 1), 400, memory_size=8)(X)
    X = svdf1rank_layer((input_size[0], 400, 1), 400, memory_size=8)(X)

    print(X.shape)
    X = Flatten()(X)
    X= Dense(out_size, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    return model
            

# test of functionalities
if __name__ == "__main__":

    # input shapes
    input_size = 40
    frames = 97
    m = 1000
    m_test= 150

    # creating random tensors for train and test
    X_train = np.random.randn(m, frames, input_size, 3)
    Y_train = np.random.randint(0, 4, m)

    X_test = np.random.randn(m_test, frames, input_size, 3)
    Y_test = np.random.randint(0, 4, m_test)
   
    # testing parameters
    pooling_size = (2, 2)
    stride = (1, 1)
    kernel  = (3, 3)
    filters = [32, 64, 128, 256, 128]
    hidden_layers = 5
    dropout_prob=0.3
    
    model = ASRModel(architecture='cnn-trad-fpool3', input_size=(frames, input_size, 3), out_size=5, pooling_size=pooling_size, \
        stride=stride, kernel=kernel, filters=filters, hidden_layers=hidden_layers, dropout_prob=dropout_prob)

    cnn_model = model.architecture

    cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print(X_train.shape, Y_train.shape)
    # training
    cnn_model.fit(x = X_train, y = Y_train, epochs=1, batch_size=7)

    # evaluate the model with test set
    preds = cnn_model.evaluate(X_test, Y_test)

    date = model.save()
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    # cnn_model = load('models/' + str(date))
    
    # print the summary of the layers(parameters)
    cnn_model.summary()
    print(cnn_model.to_json())