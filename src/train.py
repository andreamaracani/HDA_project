import argparse
import tensorflow as tf
import util as u
from model import ASRModel
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
    parser.add_argument('--class_test_samples',     type=int,       default=200,                help='Number of test samples per each class')
    parser.add_argument('--training_percentage',    type=float,     default=0.7,                help='Percentage of the dataset used for training')

    # noise samples creation
    parser.add_argument('--noise_source_path',      type=str,       default='files/',           help='Path of the noise source')
    parser.add_argument('--noise_output_path',      type=str,       default='data/26 silence/', help='Number of test samples per each class')
    parser.add_argument('--noise_samples',          type=int,       default=5000,               help='Number of noise samples to create')
    parser.add_argument('--seed',                   type=int,       default=30,                 help='Seed used for training set creation')

    # features extraction
    parser.add_argument('--pre_emphasis_coef',      type=float,     default=0.95,                help='Percentage of the dataset used for training')
    parser.add_argument('--frame_length',           type=int,       default=400,                 help='Seed used for training set creation')
    parser.add_argument('--frame_step',             type=int,       default=160,                 help='Seed used for training set creation')
    parser.add_argument('--target_frame_number',    type=int,       default=110,                 help='Seed used for training set creation')
    parser.add_argument('--random_time_shift',      type=bool,      default=True,                 help='Seed used for training set creation')
    parser.add_argument('--smooth',                 type=bool,      default=True,                 help='Seed used for training set creation')
    parser.add_argument('--smooth_length',          type=int,       default=5,                 help='Seed used for training set creation')

    # MEL Coeffs
    parser.add_argument('--hertz_from',         type=int,           default=300,                 help='Seed used for training set creation')
    parser.add_argument('--number_of_filters',  type=int,           default=40,                 help='Seed used for training set creation')
    parser.add_argument('--power_of_2',         type=bool,          default=True,                 help='Seed used for training set creation')
    parser.add_argument('--use_dct',            type=bool,          default=False,                 help='Seed used for training set creation')
    parser.add_argument('--add_delta',          type=bool,          default=True,                 help='Seed used for training set creation')
    parser.add_argument('--normalization_method', type=int,          default=0,                 help='0 no normalization, 1 standardization, 2 normalization')

normalization_method
    # augmentation
    parser.add_argument('--exclude_augmentation',      type=bool,       default=True,                 help='Seed used for training set creation')
    parser.add_argument('--augmentation_folder',       type=str,        default='augmentation',                 help='Seed used for training set creation')

    # Network arguments
    parser.add_argument('--architecture',   type=str,   default='cnn_trad_fpool3',      help="Architecture of the model to use")
    parser.add_argument('--filters',        type=int,   default=[128, 64], nargs="+",   help='Number of filters per layer')
    parser.add_argument('--kernel',         type=int,   default=[2, 2], nargs="+",      help='Kernel_size')
    parser.add_argument('--stride',         type=int,   default=[1, 1], nargs="+",      help='Stride of the kernel')
    parser.add_argument('--pool',           type=int,   default=[1,1], nargs="+",       help='Pool size')
    parser.add_argument('--hidden_layers',  type=int,   default=2,                      help='Number of convolutional layers stacked')
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

    # frames = 97
    # coeffs = 40
    # channels = 3

    # full dataset parameters
    mean_static = 5.894829478708612
    mean_delta = 0.011219158052764553
    mean_delta2 = -0.0011225424307293912
    std_static = 6.498562130522682
    std_delta = 0.6461172437961028
    std_delta2 = 0.24652535283769192
    max_static = 25.442123413085938
    min_static = -36.04365158081055
    max_delta = 16.69818878173828
    min_delta = -15.686269760131836
    max_delta2 = 9.076530456542969
    min_delta2 = -8.840045928955078

    
    normalization = args.normalization_method

    if(normalization == 0):

        # no transformation
        shift_static = 0
        scale_static = 1
        shift_delta = 0
        scale_delta = 1
        shift_delta_delta = 0
        scale_delta_delta = 1
    elif(normalization == 1):

        #standardisation
        shift_static = -mean_static
        scale_static = 1/std_static
        shift_delta = -mean_delta
        scale_delta = 1/std_delta
        shift_delta_delta = -mean_delta2
        scale_delta_delta = 1/std_delta2

    elif (normalization == 2):

        #normalization
        shift_static = -min_static
        scale_static = 1 / (max_static - min_static)
        shift_delta = -min_delta
        scale_delta = 1 / (max_delta - min_delta)
        shift_delta_delta = -min_delta2
        scale_delta_delta = 1 / (max_delta2 - min_delta2)

    # class_names = ['00 zero', '01 one','02 two','03 three','04 four','05 five','06 six',\
    #         '07 seven','08 eight','09 nine','10 go','11 yes','12 no','13 on','14 off','15 forward',\
    #         '16 backward','17 left','18 right','19 up','20 down','21 stop','22 visual','23 follow',\
    #         '24 learn','26 unknown','25 silence']

    class_names = ['00 zero', '01 one']
    num_classes = len(class_names)

    pool=tuple(args.pool)
    stride=tuple(args.stride)
    kernel=tuple(args.kernel)


    X_train, X_val, X_test, Y_train, Y_val, Y_test = \
    u.create_dataset_and_split("data/",
                               class_names=class_names,
                               training_percentage=args.training_percentage,
                               validation_percentage=1-args.training_percentage,
                               test_percentage=None,
                               training_samples=None,
                               validation_samples=None,
                               test_samples=args.class_test_samples,

                               pre_emphasis_coef=args.pre_emphasis_coef,
                               frame_length=args.frame_length,
                               frame_step=args.frame_step,
                               window_function=np.hamming,
                               target_frame_number=args.target_frame_number,
                               random_time_shift=args.random_time_shift,
                               smooth=args.smooth,
                               smooth_length=args.smooth_length,

                               hertz_from=args.hertz_from,
                               hertz_to=None,
                               number_of_filters=args.number_of_filters,
                               power_of_2=args.power_of_2,
                               dtype='float32',
                               use_dct=args.use_dct,
                               add_delta=args.add_delta,

                               # NORMALIZATION
                               shift_static=shift_static,
                               scale_static=scale_static,
                               shift_delta=shift_delta,
                               scale_delta=scale_delta,
                               shift_delta_delta=shift_delta_delta,
                               scale_delta_delta=scale_delta_delta,

                               exclude_augmentation=args.exclude_augmentation,
                               augmentation_folder=args.augmentation_folder,

                               print_info=True)

    # retrieve input size
    input_size = X_train.shape[1:]

    # initialize the network
    model = ASRModel(architecture=args.architecture, 
                    input_size=input_size, 
                    out_size=num_classes, 
                    pooling_size=pool,
                    stride=stride, 
                    kernel=kernel, 
                    filters=args.filters, 
                    hidden_layers=args.hidden_layers, 
                    dropout_prob=args.dropout_prob)

    # setting training
    model.architecture.compile(optimizer="adam", 
                                loss="categorical_crossentropy", 
                                metrics=["accuracy"])


    # passing to one-hot encoded
    Y_train = to_categorical(Y_train, num_classes)
    print(Y_train.shape)
    Y_val = to_categorical(Y_val, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    # training
    out_dir = args.ckp_folder
    # out_dir = os.path.join(os.getcwd(), out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_dir = os.path.join(out_dir, 'ckp')

    checkpoint = ModelCheckpoint(out_dir, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callbacks = [checkpoint]
    
    # earlystopping
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks.append(earlystopping)

    history = model.architecture.fit(x = X_train, y = Y_train, epochs=args.num_epochs, batch_size=args.batchsize, validation_data=(X_val, Y_val), callbacks=callbacks)


    model.architecture.summary()
    
    print("#######################")
    print("Evaluating the model")

    # evaluate with validation

    # preparing the callback for confusion matrix
    y_predict = model.architecture.predict(X_test)
    y_true = Y_test
    confusion_matrix = np.zeros((num_classes, num_classes))

    y_predict = np.argmax(y_predict, axis=1)
    y_true = np.argmax(y_true, axis=1)

    for sample in range(y_predict.shape[0]):
        confusion_matrix[y_true[sample], y_predict[sample]] = confusion_matrix[y_true[sample], y_predict[sample]] + 1

    # False Acceptance Rate = total false acceptance/total false attempts
    # where total false acceptance is the total number of non-keywords recognized as keywords.
    false_attempts = confusion_matrix[-1, :].sum()
    far = (false_attempts - confusion_matrix[-1, -1]) / false_attempts

    # False Rejection Rate - total false rejection/total true attempts
    # where total false rejection is the total number of keywords recognized as non-keyword
    frr = (confusion_matrix[:, -1].sum() - confusion_matrix[-1, -1])/ confusion_matrix[:-1, :-1].sum().sum()


    
    preds = model.architecture.evaluate(x=X_test, y=Y_test)


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
    # training_logs_folder = os.path.join(os.getcwd(), training_logs_folder)
    if not os.path.isdir(training_logs_folder):
        os.makedirs(training_logs_folder)
    filepath = os.path.join(training_logs_folder, 'training'+str(date)+'.json')

    # building data:
    training_data = {'params':vars(args),'model':model.architecture.to_json(), \
        'loss':history.history['loss'], 'acc':history.history['acc'], 
            'val_loss':history.history['val_loss'], 'val_acc':history.history['val_acc'], \
            'test_loss':str(preds[0]), 'test_acc':str(preds[1]),\
            'far':far, 'frr':frr, 'confusion_matrix':confusion_matrix.tolist()}

    with open(filepath, 'w') as f:
        json.dump(training_data, f, indent=4)

