import argparse
import tensorflow as tf
from model import ASRModel
from tensorflow.keras.callbacks import ModelCheckpoint

# parse input arguments
parser = argparse.ArgumentParser(description='Train the Automatic Speech Recognition System')

##############################
## PARAMETERS
##############################

# Dataset arguments
parser.add_argument('--datasetpath',    type=str,   default='data.npy',         help='Path of the dataset')

# Network arguments
parser.add_argument('--architecture',   type=str,   default='cnn_trad_fpool3',      help="Architecture of the model to use")
parser.add_argument('--filters',        type=int,   default=[128, 64], nargs="+",   help='Number of filters per layer')
parser.add_argument('--kernel',         type=int,   default=[2, 2], nargs="+",      help='Kernel_size')
parser.add_argument('--stride',         type=int,   default=[1, 1], nargs="+",      help='Stride of the kernel')
parser.add_argument('--pool',           type=int,   default=[1,1], nargs="+",       help='Pool size')
parser.add_argument('--hidden_layers',  type=int,   default=2,                      help='Number of convolutional stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.3,                    help='Dropout probability')

# Training argumenrs
parser.add_argument('--batchsize',      type=int,   default=154,                help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=1000,               help='Number of training epochs')

# Save arguments
parser.add_argument('--ckps_dir',     type=str,   default='model',    help='Where to save models and params')


import numpy as np
if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()

    frames = 97
    coeffs = 12

    pool=tuple(args.pool)
    stride=tuple(args.stride)
    kernel=tuple(args.kernel)

    # initialize the network
    model = ASRModel(architecture=args.architecture, input_size=(frames, coeffs, 3),  pooling_size=pool, \
        stride=stride, kernel=kernel, filters=args.filters, hidden_layers=args.hidden_layers, dropout_prob=args.dropout_prob)



    # setting training
    model.architecture.compile(optimizer="adam", 
                                loss="sparse_categorical_crossentropy", 
                                metrics=["accuracy"])

    # TODO: loading the dataset
    # X_train = pass
    # Y_train = pass

    # creating random tensors for train and test
    X_train = np.random.randn(1000, 97, 12, 3)
    Y_train = np.random.randint(0, 4, 1000)

    X_test = np.random.randn(150, 97, 12, 3)
    Y_test = np.random.randint(0, 4, 150)

    # training
    checkpoint = ModelCheckpoint(args.ckps_dir, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=15)
    history = model.architecture.fit(x = X_train, y = Y_train, epochs=args.num_epochs, batch_size=args.batchsize, callbacks=[checkpoint])

    print(history)
    # evaluate
    pred = model.architecture.evaluate(X_test, Y_test)


