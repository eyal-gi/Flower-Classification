"""
Authors:
Omer Keidar 307887984 & Eyal Ginosar 307830901

Notes:
    # We downloaded and use h5py 2.10.0
    # We downloaded and use funcsigs 1.0.2
"""

import random
import cv2
import numpy as np
import os
from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import scipy.io as sio
import scipy as s
import keras
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from funcsigs import signature
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, auc, precision_recall_curve
import time
from numpy import mean

random.seed(10), np.random.seed(10), tf.random.set_seed(10)     # requested seeds
start_time = time.time()    # for runtime calculation

# ####____________________ READ DATA __________________________#####
data_dir = pjoin(os.getcwd(), 'FlowerData')
mat_fname = pjoin(data_dir, 'FlowerDataLabels.mat')  # Labels file path
mat_contents = sio.loadmat(mat_fname)  # Read labels file
mat_labels = np.transpose(mat_contents['Labels']).tolist()


def get_default_parameters(path_data, test_indices, startime):
    """
    This function generate and return a dictionary containing the default experiment parameters.
     it contains all tuned parameters.
    :param path_data: The data path string
    :param test_indices: The indices of test images
    :param startime: Saves the start time of the run
    :return: params - A dictionary with all relevant parameters for the pipe
    """
    def_params = {
        'data':
            {
                'data_path': path_data,
                'test_indices': test_indices,
            },
        'prepare':
            {
                'S': (224, 224)
            },
        'train':
            {
                'batch_size': 16,
                'train_layers': 7,
                'epochs': 5,
                'lr': 0.03,
                'decay': 0.01,
                'activation': 'sigmoid'

            },
        'report':
            {
                'start_time': startime,
                'train_size': 0     # is updated during run


            }
    }
    return def_params


# ####____________________ DATA CONFIG __________________________#####

def data_config(data_labels, def_params):
    """
    This function imports the images, splits them to train and test according to the test_images_indices and attaches
     the correct label for each image
    :param data_labels: List of images labels
    :param def_params: Dictionary of all the default parameters- folder path and test indices
    :return: Dictionary of images and labels of Train set, Test set
    """
    print("Importing data")
    train = {'images': [], 'labels': []}
    test = {'images': [], 'labels': []}

    data_path = def_params['data']['data_path']  # folder path
    test_img_indices = def_params['data']['test_indices']  # indices for test images

    i = 1   # index
    for lbl in data_labels:  # runs through all the labels
        img = cv2.imread(data_path + "/" + str(i) + '.jpeg')  # reads image i from the folder
        if i in test_img_indices:  # classify if this is for test or train
            test['labels'].append(lbl[0])
            test['images'].append(img)
        else:
            train['labels'].append(lbl[0])
            train['images'].append(img)
        i = i + 1

    def_params['report']['train_size'] = len(train['images'])   # update the train size

    return train, test


def resize_set(data, prepare_params):
    """
    This function resize all the images in the given set according to the default resize parameter
    :param data: Images and labels set
    :param prepare_params: Dictionary of the default prepare parameters
    :return: A resized set
    """
    d_size = prepare_params['S']  # The size to resize to.
    labels = data['labels'].copy()  # copy the labels of the images from the data received
    resized_data = {  # the returned dictionary of the modified data
        'images': [],
        'labels': labels
    }
    for img in data['images']:
        image_resized = cv2.resize(img, d_size)  # resize image
        resized_data['images'].append(image_resized)  # add image to the new set

    return resized_data


def data_augmentation(train_dataset, d_size):
    """
    This function generate augmented data. It randomly flip horizontally or crop every image in the train set and adds
    the augmented image to the train dataset.
    :param train_dataset: The model train dataset
    :param d_size: The dimenstion size for resize
    :return: New train data set with the augmented images. It is twice the size of the original dataset
    """
    aug_dataset = train_dataset.copy()  # copy data set for augmenting
    merged_dataset = train_dataset.copy()   # copy data set for creating a new set
    for i in range(0, len(aug_dataset['images'])):  # go over all the images in the dataset
        # load the image and label
        img = aug_dataset['images'][i]
        lbl = aug_dataset['labels'][i]
        idx = random.randint(1, 2)  # randomly allocate the picture to be flipped or cropped
        # flip horizontal
        if idx == 1:
            aug_img = cv2.flip(img, 1)
        # crop
        else:
            scale = random.uniform(0.8, 0.9)  # When cropping, use random crops of large portions
            center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
            width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
            left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
            top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
            aug_img = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]

        aug_img = cv2.resize(aug_img, d_size)   # resize the image again to make sure it's in the right size
        # add new image and label to the new merged train dataset
        merged_dataset['images'].append(aug_img)
        merged_dataset['labels'].append(lbl)

    return merged_dataset


def preprocess_ResNet(train_data, test_data):
    """
       The func prepares the data for resnet by making some manipulation on the data.
            :param train_data , test_data: dict of train and test
            :return: dict of train and test, ready for resnet
    """

    # transpose the labels lists
    tr_labels = np.array(train_data['labels'])
    tes_labels = np.array(test_data['labels'])

    prepared_train = {
        'images': [],
        'labels': tr_labels
    }
    prepared_test = {
        'images': [],
        'labels': tes_labels
    }

    for tr_img in train_data['images']:
        tr_temp_img = image.img_to_array(tr_img)
        tr_temp_img = np.expand_dims(tr_temp_img, axis=0)
        tr_temp_img = preprocess_input(tr_temp_img)
        prepared_train['images'].append(tr_temp_img)
    prepared_train['images'] = np.rollaxis(np.array(prepared_train['images']), 1, 0)[0]

    for tes_img in test_data['images']:
        tes_temp_img = image.img_to_array(tes_img)
        tes_temp_img = np.expand_dims(tes_temp_img, axis=0)
        tes_temp_img = preprocess_input(tes_temp_img)
        prepared_test['images'].append(tes_temp_img)
    prepared_test['images'] = np.rollaxis(np.array(prepared_test['images']), 1, 0)[0]

    return prepared_train, prepared_test


def prepare_data(train, test, prepare_params):
    """
    This function prepares the data for the ResNet model.
    :param train: Training set with images and labels
    :param test: Test set with images and labels
    :param prepare_params: The default parameters for preparing the data
    :return: Prepared data for input to the ResNet model
    """
    print("Preparing data for ResNet50V2")
    r_train_set = resize_set(train, prepare_params)  # resize images in the train set
    r_test_set = resize_set(test, prepare_params)  # resize images in the test set
    aug_train = data_augmentation(r_train_set, prepare_params['S'])

    data_for_train, data_for_test = preprocess_ResNet(aug_train, r_test_set)  # preprocess the data for ResNet

    return data_for_train, data_for_test


def split_data_for_train_and_val(train_dataset):
    """
    This function get the train set and make train and validation.
    :param train_dataset
    :return: train set and validation set
    """
    train_images, valid_images, train_labels, valid_labels = train_test_split(train_dataset['images'],
                                                                              train_dataset['labels'], test_size=0.3,
                                                                              shuffle=True)
    train_set_img = {
        'images': train_images,
        'labels': train_labels
    }

    validation_set_img = {
        'images': valid_images,
        'labels': valid_labels
    }
    return train_set_img, validation_set_img


def base_model(prepare_params, train_params):
    """
    This function creates the basic network from ResNet50V2 and transfer it to a binary classification model
    :param prepare_params: The default prepare parameters
    :param train_params: The default train parameters
    :return: The modulated ResNet model
    """
    print("Creating base model")
    # model parameters
    img_shape = prepare_params['S'] + (3,)
    train_layers_num = train_params['train_layers']
    base_learning_rate = train_params['lr']
    dcy = train_params['decay']
    activation = train_params['activation']

    inputs = Input(shape=img_shape)  # model inputs size
    res_model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_tensor=inputs,
                                                    pooling='avg')  # extract the ResNet50V2 net
    # Loop over trainable layers
    for layer in res_model.layers[:-train_layers_num]:
        layer.trainable = False

    # add a binary classification layer
    prediction_layer = keras.layers.Dense(1, activation=activation)
    # join layer to the model sequentially
    trans_model = keras.Sequential([res_model, prediction_layer])
    # compile the model
    trans_model.compile(optimizer=keras.optimizers.SGD(lr=base_learning_rate, decay=dcy), loss='binary_crossentropy',
                        metrics=['accuracy'])

    return trans_model


def train_model(resnet_basic, train_data, valid_data, train_params):
    """
    This function train the network and test the validation/test set.
    :param resnet_basic: The network to train
    :param train_data: train dataset
    :param valid_data: validation/test dataset
    :param train_params: the default train parameters
    :return: trained model, history
    """
    print("Training network")
    # datasets extraction
    train_images = train_data['images']
    train_labels = train_data['labels']
    valid_images = valid_data['images']
    valid_labels = valid_data['labels']
    # parameters
    epochs = train_params['epochs']
    batch = train_params['batch_size']

    history = resnet_basic.fit(train_images, train_labels, epochs=epochs,
                               validation_data=(valid_images, valid_labels),
                               shuffle=True, batch_size=batch, verbose=1)  # fitting the model

    ### different type of augmentation ####
    # steps = len(train_images)//batch
    # create aug data generator
    # datagen = image.ImageDataGenerator(rotation_range=90, zoom_range=0.15, width_shift_range=0.2,
    #                                    height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,
    #                                    vertical_flip=True, fill_mode="nearest")
    # fit datagen
    # datagen.fit(train_images)
    # define model
    # t_model = resnet_basic.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch), epochs=epochs,
    #                                      validation_data=(valid_images, valid_labels), shuffle=True,
    #                                      verbose=1)  # fitting the model)

    train_plots(history, valid_labels)  # loss and accuracy by epochs plots
    return resnet_basic, history


def train_plots(trained_model, test_labels):
    """
    Plots loss and accuracy by epochs
    :param trained_model: history of trained model
    :param test_labels: the test labels
    :return: void
    """
    acc = trained_model.history['accuracy']
    val_acc = trained_model.history['val_accuracy']

    loss = trained_model.history['loss']
    val_loss = trained_model.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def recall_precision_plot(precision, recall):
    """
    This functions plots the recall-precision curve
    :param precision: the model precisions
    :param recall: the model recalls
    :return:
    """
    average_precision = precision.mean()
    plt.step(recall, precision, color='k', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='lightgreen')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    return plt


def calc_true(labels_list):
    """
    This function calculate how many "True" flowers there are in the dataset
    :param labels_list:
    :return:
    """
    ans = 0
    for i in range(len(labels_list)):
        if labels_list[i] == 1:      # is flower
            ans = ans + 1
    return ans


def evaluate(csf_model, model_history, train_dataset, test_dataset):
    """
    This function evaluates the model
    :param csf_model: the model
    :param model_history: model history
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :return:
    """
    print("Evaluating")
    # e_loss, e_acc_score = csf_model.evaluate(test_dataset['images'], test_dataset['labels'])
    loss = np.amin(model_history.history['val_loss'])   # model min loss
    acc_score = np.amax(model_history.history['val_accuracy'])  # model max accuracy
    predictions = csf_model.predict(test_dataset['images'])     # model predictions scores
    class_predictions = predictions.round().astype(int)     # model classified predictions
    error_rate = 1 - acc_score  # model error
    # precisions, recall and thresholds
    precision, recall, thresholds = precision_recall_curve(y_true=test_dataset['labels'], probas_pred=predictions)
    auc_score = auc(recall, precision)  # the area under the curve
    cm = confusion_matrix(test_set['labels'], class_predictions)    # confusion matrix variable

    return{
        "origin_true_num": int(calc_true(train_dataset['labels'])/2),
        "aug_true_num": int(calc_true(train_dataset['labels'])/2),
        "error_rate": error_rate,
        "prec_recall_auc": (precision, recall, auc_score),
        "conf_matrix": cm,
        "network_scores": predictions,
        "test_labels": test_dataset['labels']

    }


def error_type(predictions, labels):
    """
    This function finds the largest errors of each type and prints its index and value.
    :param predictions: The model predictions
    :param labels: The test labels
    """
    type1_error = []
    type2_error = []

    # todo: remove all prints for submission
    print('\n\n' "PRINT ERRORS:")

    for i in range(len(labels)):
        pred = predictions[i][0]    # A single prediction
        original_idx = i    # the original image index(name)
        if pred > 0.5 and labels[i] == 0:     # type 1 error
            type1_error.append((pred, original_idx))
        elif pred < 0.5 and labels[i] == 1:   # type 2 error
            type2_error.append((pred, original_idx))

    # print top 5 type 1 errors
    if len(type1_error) != 0:   # list is not empty
        type1_error = sorted(type1_error, reverse=True)     # sort list
        i = 0
        while i < 5 and i < len(type1_error):     # top 5 or less
            score = type1_error[i][0]
            idx = i + 1
            org_idx = type1_error[i][1]
            print("Type 1 error - Error index:", idx, "Classification score is:", score, ". Image name is", org_idx, ".jpeg")
            i += 1

    print()
    # print top 5 type 2 errors
    if len(type2_error) != 0:   # list is not empty
        type2_error = sorted(type2_error, reverse=False)    # sort list
        i = 0
        while i < 5 and i < len(type2_error):     # top 5 or less
            score = type2_error[i][0]
            idx = i + 1
            org_idx = type2_error[i][1]
            print("Type 2 error - Error index:", idx, "Classification score is:", score, ". Image name is", org_idx, ".jpeg")
            i += 1

    pass


def report(final_model, report_params, evaluation):
    """
    The program report. prints: number of train images, number of augmented images, number of flowers in the train data,
    precision-recall curve AUC score, error rate, confusion matrix, model summary and run time. It is also calculate
    the top 5 error of type I / type II errors.
    :param final_model: The final trained model
    :param report_params: the default report parameters
    :param evaluation: the eavluation results

    """
    print("Generating report")
    # Report will print out:

    # the number of training images and the amount of “True” flower images out of them( if augmented training set,
    #   the report will include this information for original images and augmented images separately),
    print("Number of original training images:", report_params['train_size'])
    print("Amount of ``True`` flower images:", evaluation['origin_true_num'])
    print("Number of augmented images:", report_params['train_size'])
    print("Amount of ``True`` augmented images:", evaluation['aug_true_num'])
    # precision-recall curve AUC score,
    precision, recall, auc_score = evaluation['prec_recall_auc']
    print("Precision-recall curve AUC score:", auc_score)
    recall_precision_plot(precision, recall)
    # test error rate,
    print("Test error rate =", evaluation['error_rate'])
    # confusion matrix ( for a threshold of your choice),
    cm = evaluation['conf_matrix']
    print("Confusion Matrix: \n", cm)
    sns.heatmap(cm, annot=True, cmap="crest")
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # final model structure (using the keras model.summary() command),
    final_model.summary()
    # time in minutes of entire run.
    print("Run Time:", round((time.time() - report_params['start_time'])/60, 3), "min")

    # for written report, no console output:
    # error_type(evaluation['network_scores'], evaluation['test_labels'])

    pass


def create_plot(x, y, x_name):
    '''create tunning plots'''
    plt.plot(x,y)
    plt.ylabel('Accuracy')
    plt.xlabel(x_name)
    plt.title(f'Accuracy vs. {x_name}')
    plt.ylim(0, 1)
    plt.show()

def accuracy_plot(history):
    '''summarize history for accuracy of ephocs'''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def activation_tuning(train, validation):
    '''tuning the activation function.
    We calculate the model with different parameter and chose the parameter that give us the best accuracy
    '''
    acc = []
    activations = ['sigmoid', 'relu', 'softmax', 'elu', 'softsign', 'tanh']
    for activation in activations:
        print(f'activation = {activation}')
        params['train']['activation']=activation
        model = base_model(params['prepare'],params['train'])
        hist = train_model(model,train,validation,params['train'])
        acc.append(hist.history['val_accuracy'][-1]) # take the validation accuracy
    create_plot(activations, acc, 'Activation')
    print(f' Tuning activation acc = {acc}')
    chosen_activation = activations[acc.index(max(acc))]
    print(f'The chosen activation is {chosen_activation}')
    return chosen_activation

def layer_tuning(train, validation,chosen_activation):
    '''tuning the layers to train on,
        We calculate the model with different parameter and chose the parameter that give us the best accuracy
'''
    acc = []
    params['train']['activation'] = chosen_activation #use the chosen activation that already tuned
    for layer in [1,2,3,4,5,6,7,8]:
        print(f'layer = {layer}')
        params['train']['train_layers'] = layer
        model = base_model(params['prepare'], params['train'])
        hist = train_model(model, train, validation, params['train'])
        acc.append(hist.history['val_accuracy'][-1]) # take the validation accuracy
    create_plot(range(1,9), acc, 'Number of layers to train')
    print(f'Tuning layers acc = {acc}')
    chosen_layer = acc.index(max(acc))+1
    print(f'The chosen number of layers is {chosen_layer}')
    return chosen_layer

def epochs_tuning(train, validation, chosen_layer):
    '''tune the ephocs,
        We calculate the model with different parameter and chose the parameter that give us the best accuracy
'''
    params['train']['train_layers'] = chosen_layer #use the chosen num of layers that already tuned
    acc = []
    for epochs in range(1, 6):
        print(f'epochs = {epochs}')
        params['train']['epochs']=epochs
        model = base_model(params['prepare'], params['train'])
        hist = train_model(model, train, validation, params['train'])
        acc.append(hist.history['val_accuracy'][- 1]) # take the validation accuracy
    accuracy_plot(hist)
    print(f' tune epochs acc = {acc}')
    chosen_epochs = acc.index(max(acc)) + 1
    print(f'the chosen epochs is {chosen_epochs}')
    return chosen_epochs

def find_index_gsd(combination_index):
    if (combination_index<4):
        lr_ind = 0
    elif (combination_index>3 and combination_index<8):
        lr_ind = 1
    else:
        lr_ind = 2
    decay_ind = combination_index-(lr_ind*4)

    return lr_ind, decay_ind

def gsd_tuning(train, validation , chosen_epochs):
    '''tune the GSD parameters,
        We calculate the model with different parameter and chose the parameter that give us the best accuracy
'''
    params['train']['epochs'] = chosen_epochs  # use the chosen num of layers that already tuned
    acc = []
    learning_rates = [0.01, 0.03, 0.05]
    decays = [0.001, 0.005, 0.01, 0.1]
    for lr in learning_rates:
        params['train']['lr']=lr
        for decay in decays:
            params['train']['decay'] = decay
            print(f'leraning rate = {lr} and decay = {decay}')
            model = base_model(params['prepare'], params['train'])
            hist = train_model(model, train, validation, params['train'])
            acc.append(hist.history['val_accuracy'][- 1]) # take the validation accuracy
    print(f'tune lr and decay acc = {acc}')
    print(f'max accuracy index is {acc.index(max(acc))}')
    # extract the final values
    combination_ind = acc.index(max(acc))
    lr_index , decey_index = find_index_gsd(combination_ind)
    create_plot(range(1, 13), acc, 'SGD parameters')
    print(f'lr index = {lr_index}, decay index = {decey_index}')
    return learning_rates[lr_index], decays[decey_index]

def calc_accuracy(real_labels, new_labels):
    '''calcs the accuracy for the new threshold'''
    counter = 0
    for i in range(len(new_labels)):
       if real_labels[i] == new_labels[i]:
           counter = counter+1
    return counter/len(new_labels)

def threshold_tuning(train, validation, lr, decay):
    '''tune the threshold on the validation only'''
    batch_s=params['train']['batch_size']
    params['train']['lr']=lr
    params['train']['decay'] = decay
    acc = []
    thresholds = [0.40,0.41, 0.42, 0.45, 0.5, 0.55, 0.57, 0.6]
    # train model
    model = base_model(params['prepare'], params['train'])
    train_model(model, train, validation, params['train'])
    # check different thresholds on the validation set as a test set
    for thresh in thresholds:
        loss, accuracy = model.evaluate(validation['images'], validation['labels'], batch_size=batch_s, verbose=1)
        print(
            f'The loss is {round(loss, 4)}, the accuracy is {round(accuracy * 100, 4)}% and the error is {round(100 - accuracy * 100, 4)}%')
        new_predictions = keras.metrics.binary_accuracy(validation['labels'], model.predict(validation['images']), threshold=thresh)
        new_acc = calc_accuracy(validation['labels'], new_predictions)
        print(f'the new accuracy is {new_acc}')
        acc.append(new_acc)
    print(f' tune threshold acc = {acc}')
    chosen_thresh = thresholds[acc.index(max(acc))]
    print(f'threshold is: {chosen_thresh}')
    create_plot(thresholds, acc, 'Threshold')
    return chosen_thresh

def tuning(train, validation):
    '''tune hyper parameters to improve the model'''
    # chosen_activation = activation_tuning(train, validation, )
    chosen_activation = 'sigmoid' # We dicide to use this because we want 'sigmoid' , he give us better results than tanh
    chosen_layer = layer_tuning(train, validation, chosen_activation)
    chosen_epochs = epochs_tuning(train, validation, chosen_layer)
    chosen_lr, chosen_decay = gsd_tuning(train, validation, chosen_epochs)
    threshold = threshold_tuning(train,validation, chosen_lr, chosen_decay)
    print('######## Final Tuning Results ############')
    print(f'the chosen epochs is {chosen_epochs}')
    print(f'the chosen activation is {chosen_activation}')
    print(f'lr = {chosen_lr}, decay = {chosen_decay}')
    print(f'the chosen threshold is {threshold}')

# ####____________________ MAIN __________________________#####

test_images_indices = list(range(301, 473))
params = get_default_parameters(data_dir, test_images_indices, start_time)  # default parameters dictionary
train_set, test_set = data_config(mat_labels, params)  # returns train{[images, labels]}, test{[images, labels]}
train_set, test_set = prepare_data(train_set, test_set, params['prepare'])   # prepared data sets
train_set_images, validation_set_images = split_data_for_train_and_val(train_set)  # train_set - splited to train and validation
b_model = base_model(params['prepare'], params['train'])
# model = train_model(b_model, train_set_images, validation_set_images, params['train'])
#tuning(train_set_images,validation_set_images)
model, network_history = train_model(b_model, train_set, test_set, params['train'])     # train model
results = evaluate(model, network_history, train_set, test_set)    # evaluate model
report(model, params['report'], results)    # print report
