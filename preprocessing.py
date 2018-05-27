#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 09:00:50 2018

@author: longtran
"""

import numpy as np
import os
import tensorflow as tf


def load_data(data_dir, input_file, media_label_file, emotion_label_file):
    """
    Load the dataset (including the input arrays and media/emotion labels) and
    perform preprocessing on the input arrays
    
    Inputs:
        - data_dir: the directory where the dataset is located
        - input_file: the name of the npz file that stores input arrays
        - media_label_file: the name of the npz file that stores media labels
        - emotion_label_file: the name of the npz file that stores emotion labels
    
    Returns:
        - train, dev, test, each of which is a tuple (input arrays, media labels, emotion labels)
    """
    
    
    # Load the BAM dataset from .npz files and use appropriate data types and shapes
    input_data = np.load(os.path.join(data_dir, input_file))
    media_data = np.load(os.path.join(data_dir, media_label_file))
    emotion_data = np.load(os.path.join(data_dir, emotion_label_file))
    
    # Read the train, dev, and test inputs together with their labels
    X_train = input_data['train']
    X_val = input_data['dev']
    X_test = input_data['test']
    
    y_media_train = media_data['train']
    y_media_val = media_data['dev']
    y_media_test = media_data['test']
    y_emotion_train = emotion_data['train']
    y_emotion_val = emotion_data['dev']
    y_emotion_test = emotion_data['test']
    
    # Print the shapes for potential debugging
    print('Train data shape:', X_train.shape)
    print('Media train labels shape:', y_media_train.shape, y_media_train.dtype)
    print('Emotion train labels shape:', y_emotion_train.shape, y_emotion_train.dtype)
    print('Validation data shape:', X_val.shape)
    print('Media validation labels shape:', y_media_val.shape, y_media_val.dtype)
    print('Emotion validation labels shape:', y_emotion_val.shape, y_emotion_val.dtype)
    print('Test data shape:', X_test.shape)
    print('Media test labels shape:', y_media_test.shape, y_media_test.dtype)
    print('Emotion test labels shape:', y_emotion_test.shape, y_emotion_test.dtype)
    
    # Normalize validation and test data: subtract the mean pixel and divide by std
    # from the train data (Note that the normalization of the train data happens
    # through ImageDataGenerator)
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel
    
    
    # Return tuples of train, dev, test
    train_data = (X_train, y_media_train, y_emotion_train)
    val_data = (X_val, y_media_val, y_emotion_val)
    test_data = (X_test, y_media_test, y_emotion_test)
    
    return train_data, val_data, test_data
    
def preprocess(train_data, augment=False):   
    X_train = train_data[0]
    if augment:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                                        featurewise_std_normalization=True,
                                                                        zca_whitening=True,
                                                                        shear_range=0.2,
                                                                        zoom_range=0.0,
                                                                        horizontal_flip=False,
                                                                        vertical_flip=True)
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                                        featurewise_std_normalization=True)
    
    train_datagen.fit(X_train)
    return train_datagen

def gen_generator(generator, X, y_media, y_emotion, batch_size, seed=7, num_classes_media=7):
    combine_labels = np.concatenate((y_media, y_emotion), axis=1) 
    generator = generator.flow(X, combine_labels, batch_size=batch_size, seed=seed)
    while True:
        X_batch, combine_labels_batch =  generator.next()
        y_media_batch = combine_labels_batch[:, :num_classes_media]
        y_emotion_batch = combine_labels_batch[:, num_classes_media:]
        yield (X_batch, {'output_media': y_media_batch, 'output_emotion': y_emotion_batch})
    