import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model



"""
    Create a CNN model:
    [padding] -> [Conv] -> [Batch-norm] -> [Relu] -> [MaxPool] -> [softmax]


    Args:
        input_shape: tuple of ints indicating the shape of the input
    
    Returns:
        an instance of the model created

"""
def toy_model(input_shape):

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



"""
    Create the cnn-trad-fpool3 model as Convolutional Neural Networks for Small-fooprint Keyword Spotting
    [Sainath15]:

    [Conv] -> [MaxPool] -> [Conv] -> [Linear] -> [Relu] -> [softmax]
    

    Args:
        input_shape: tuple of ints indicating the shape of the input
    
    Returns:
        an instance of the model created

"""
def model(input_shape):

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



if __name__ == "__main__":

    # input shapes
    input_size = 40
    frames = 97
    m = 3000
    m_test= 150

    # creating random tensors for train and test
    X_train = np.random.randn(m, frames, input_size, 3)
    Y_train = np.random.randint(0, 4, m)

    X_test = np.random.randn(m_test, frames, input_size, 3)
    Y_test = np.random.randint(0, 4, m_test)

    print(Y_test)
   
    # initialize the computational graph
    cnn_model = model((frames, input_size, 3))

    cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # training
    cnn_model.fit(x = X_train, y = Y_train, epochs=5, batch_size=2)

    # evaluate the model with test set
    preds = cnn_model.evaluate(X_test, Y_test)

    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
    # print the summary of the layers(parameters)
    cnn_model.summary()