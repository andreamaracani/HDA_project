import argparse
import tensorflow as tf
import util as u
from model import ASRModel
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# parse input arguments
parser = argparse.ArgumentParser(description='Train the Automatic Speech Recognition System')

##############################
## PARAMETERS
##############################

# Dataset arguments
parser.add_argument('--datasetpath',            type=str,       default='data/',            help='Path of the dataset')
parser.add_argument('--class_test_samples',     type=int,       default=150,                help='Number of test samples per each class')
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
parser.add_argument('--dropout_prob',   type=float, default=0.3,                    help='Dropout probability')

# Training argumenrs
parser.add_argument('--batchsize',      type=int,   default=64,                help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=1000,               help='Number of training epochs')

# Save arguments
parser.add_argument('--ckp_file',     type=str,   default='models/',    help='Where to save models and params')


import numpy as np
if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()

    frames = 97
    coeffs = 40

    pool=tuple(args.pool)
    stride=tuple(args.stride)
    kernel=tuple(args.kernel)

    # initialize the network
    model = ASRModel(architecture=args.architecture, input_size=(frames, coeffs, 3),  pooling_size=pool, \
        stride=stride, kernel=kernel, filters=args.filters, hidden_layers=args.hidden_layers, dropout_prob=args.dropout_prob)

    # model = ASRModel(architecture='cnn-trad-fpool3', input_size=(frames, coeffs, 3))


    # setting training
    model.architecture.compile(optimizer="adam", 
                                loss="sparse_categorical_crossentropy", 
                                metrics=["accuracy"])

    # loading the dataset
    input_path_data = args.datasetpath
    test_samples_per_class = args.class_test_samples
    training_percentage = args.training_percentage



    u.get_samples_from_noise(args.noise_source_path, args.noise_output_path, nOutput=args.noise_samples, seed=args.seed)
    dataset = u.create_dataset(input_path_data, max_files_per_class=None)

    print("#######################")
    print("Splitting the dataset")
    
    train, val, test, train_l, val_l, test_l = u.split_dataset(dataset,  test_samples_per_class, training_percentage)

    print("#######################")
    print("Starting training")

    # training
    checkpoint = ModelCheckpoint(args.ckp_file, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=15)
    history = model.architecture.fit(x = train, y = train_l, epochs=args.num_epochs, batch_size=args.batchsize, callbacks=[checkpoint])

    print("#######################")
    print("Evaluating the model")

    # evaluate with validation
    preds = model.architecture.evaluate(val, val_l)

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

