import argparse
import tensorflow as tf
import util as u
from model import ASRModel
from keras.callbacks import ModelCheckpoint, RemoteMonitor, Callback
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import keras.backend as K

def get_args():
    
    # parse input arguments
    parser = argparse.ArgumentParser(description='Train the Automatic Speech Recognition System')

    ##############################
    ## PARAMETERS
    ##############################

    # Dataset arguments
    parser.add_argument('--datasetpath',            type=str,       default='data/',            help='Path of the dataset')
    parser.add_argument('--class_test_samples',     type=int,       default=50,                help='Number of test samples per each class')
    parser.add_argument('--classes',                type=int,       default=None,                help='Number of classes used')
    parser.add_argument('--training_percentage',    type=float,     default=0.7,                help='Percentage of the dataset used for training')

    # noise samples creation
    parser.add_argument('--noise_source_path',      type=str,       default='files/',           help='Path of the noise source')
    parser.add_argument('--noise_output_path',      type=str,       default='data/26 silence/', help='Number of test samples per each class')
    parser.add_argument('--noise_samples',          type=int,       default=5000,               help='Number of noise samples to create')
    parser.add_argument('--seed',                   type=int,       default=30,                 help='Seed used for training set creation')


    # Network arguments
    parser.add_argument('--architecture',   type=str,   default='cnn_trad_fpool3',      help="Architecture of the model to use")
    parser.add_argument('--filters',        type=int,   default=[128, 64], nargs="+",   help='Number of filters per layer')
    parser.add_argument('--kernel',         type=int,   default=[2, 2], nargs="+",      help='Kernel_size')
    parser.add_argument('--stride',         type=int,   default=[1, 1], nargs="+",      help='Stride of the kernel')
    parser.add_argument('--pool',           type=int,   default=[1,1], nargs="+",       help='Pool size')
    parser.add_argument('--hidden_layers',  type=int,   default=2,                      help='Number of convolutional stacked layers')
    parser.add_argument('--dropout_prob',   type=float, default=0.25,                    help='Dropout probability')

    # Training argumenrs
    parser.add_argument('--batchsize',      type=int,   default=64,                help='Training batch size')
    parser.add_argument('--num_epochs',     type=int,   default=100,               help='Number of training epochs')

    # Save arguments
    parser.add_argument('--ckp_folder',     type=str,   default='models/',    help='Where to save models and params')
    parser.add_argument('--training_logs_folder',type=str,   default='training_logs/',    help='Where to save training logs and params')

    return parser.parse_args()

def far_metric(y_true, y_pred):
    """
    Return the FAR(False Acceptance Rate) metric which is the ratio
    total false acceptance/total false attempts.

    Where total false acceptance is the total number of non-keywords recognized as keywords.

    """
    # print(y_true)
    # print(y_pred)
    # print(K.sum(y_true, y_pred))


    # return K.sum(y_true, y_pred)[-1]
# def frr_metric(y_true, y_pred):
#     """
#     Return the FRR(False Rejection Rate) metric which is the ratio
#     total false rejection/total true attempts
#     """
#     return 

import numpy as np
if __name__ == "__main__":

    # Parse input arguments
    args = get_args()

    frames = 97
    coeffs = 40
    channels = 3

    pool=tuple(args.pool)
    stride=tuple(args.stride)
    kernel=tuple(args.kernel)

    # initialize the network
    model = ASRModel(architecture=args.architecture, input_size=(frames, coeffs, 3), out_size=args.classes, pooling_size=pool, \
        stride=stride, kernel=kernel, filters=args.filters, hidden_layers=args.hidden_layers, dropout_prob=args.dropout_prob)

    # setting training
    model.architecture.compile(optimizer="adam", 
                                loss="categorical_crossentropy", 
                                metrics=["accuracy"])

    class Metrics(Callback):

        def on_train_begin(self, logs={}):
            self._data = np.zeros((num_classes, num_classes))

        def on_epoch_end(self, batch, logs={}):
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            y_predict = np.asarray(model.architecture.predict(X_val))

            y_val = np.argmax(y_val, axis=1)
            y_predict = np.argmax(y_predict, axis=1)

            self._data[y_val, y_predict] = self._data[y_val, y_predict] + 1
            return

        def get_data(self):
            return self._data
        
    # loading the dataset
    input_path_data = args.datasetpath
    test_samples_per_class = args.class_test_samples
    training_percentage = args.training_percentage
    num_classes = args.classes


    # u.get_samples_from_noise(args.noise_source_path, args.noise_output_path, nOutput=args.noise_samples, seed=args.seed)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = u.create_dataset_and_split(input_path_data, test_samples_per_class, \
        training_percentage, (frames, coeffs, channels), max_classes=num_classes, addDelta=True)

    # passing to one-hot encoded
    Y_train = to_categorical(Y_train, num_classes)
    print(Y_train.shape)
    Y_val = to_categorical(Y_val, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    # training
    out_dir = args.ckp_folder
    out_dir = os.path.join(os.getcwd(), out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_dir = os.path.join(out_dir, 'ckp')

    checkpoint = ModelCheckpoint(out_dir, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks = [checkpoint]
    metrics = Metrics()
    callbacks.append(metrics)
    history = model.architecture.fit(x = X_train, y = Y_train, epochs=args.num_epochs, batch_size=args.batchsize, validation_data=(X_val, Y_val), callbacks=callbacks)

    print(metrics.get_data())
    confusion_matrix = metrics.get_data()

    # False Acceptance Rate = total false acceptance/total false attempts
    # where total false acceptance is the total number of non-keywords recognized as keywords.
    false_attempts = confusion_matrix[-1, :].sum()
    far = (false_attempts - confusion_matrix[-1, -1]) / false_attempts
    print(far)
    # False Rejection Rate - total false rejection/total true attempts
    # where total false rejection is the total number of keywords recognized as non-keyword
    frr = (confusion_matrix[:, -1].sum() - confusion_matrix[-1, -1])/ confusion_matrix[:-1, :-1].sum().sum()
    print(frr)


    model.architecture.summary()
    
    print("#######################")
    print("Evaluating the model")

    # evaluate with validation
    preds = model.architecture.evaluate(X_val, Y_val)


    date = model.save()

    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))


    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig('plots/accuracy.png', format='png')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig('plots/loss.png', format='png')

    # training
    training_logs_folder = args.training_logs_folder
    training_logs_folder = os.path.join(os.getcwd(), training_logs_folder)
    if not os.path.isdir(training_logs_folder):
        os.makedirs(training_logs_folder)
    filepath = os.path.join(training_logs_folder, 'training'+str(date)+'.json')

    # building data:
    training_data = {'params':vars(args),'model':model.architecture.to_json(), \
        'loss':history.history['loss'], 'acc':history.history['acc'], 'far':far, 'frr':frr, 'confusion_matrix':confusion_matrix.tolist()}

    with open(filepath, 'w') as f:
        json.dump(training_data, f, indent=4)

