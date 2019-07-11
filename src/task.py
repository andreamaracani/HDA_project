import argparse
import tensorflow as tf
import util as u
from model import ASRModel
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json

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
    
    # loading the dataset
    input_path_data = args.datasetpath
    test_samples_per_class = args.class_test_samples
    training_percentage = args.training_percentage
    num_classes = args.classes


    u.get_samples_from_noise(args.noise_source_path, args.noise_output_path, nOutput=args.noise_samples, seed=args.seed)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = u.create_dataset_and_split(input_path_data, test_samples_per_class, \
        training_percentage, (frames, coeffs, channels), max_classes=num_classes)

    # passing to one-hot encoded
    Y_train = to_categorical(Y_train, num_classes)
    Y_val = to_categorical(Y_val, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    # training
    out_dir = Path(args.ckp_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(out_dir, 'ckp')

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=2)
    # remoteMonitor = RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
    callbacks = [checkpoint]
    history = model.architecture.fit(x = X_train, y = Y_train, epochs=args.num_epochs, batch_size=args.batchsize, validation_data=(X_val, Y_val), callbacks=callbacks)

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
    training_logs_folder = Path(args.training_logs_folder)
    training_logs_folder.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(training_logs_folder, 'training'+str(date)+'.json')

    # building data:
    training_data = {'params':vars(args),'model':model.architecture.to_json(), 'loss':history.history['loss'], 'acc':history.history['acc']}

    with open(filepath, 'w') as f:
        json.dump(training_data, f, indent=4)

