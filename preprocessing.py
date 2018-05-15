#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 09:00:50 2018

@author: longtran
"""

import numpy as np
import tensorflow as tf
from data_utils import DATA_DIR, INPUT_FILE, MEDIA_LABEL_FILE, EMOTION_LABEL_FILE
import os


def preprocess(data_dir, input_file, media_label_file, emotion_label_file):
    # Load the BAM dataset from .npz files and use appropriate data types and shapes
    input_data = np.load(os.path.join(DATA_DIR, INPUT_FILE))
    media_data = np.load(os.path.join(DATA_DIR, MEDIA_LABEL_FILE))
    emotion_data = np.load(os.path.join(DATA_DIR, EMOTION_LABEL_FILE))
    
    # Read the train, dev, and test inputs together with their labels
    X_train = input_data['train']
    X_val = input_data['dev']
    X_test = input_data['test']
    y_media_train = np.argmax(media_data['train'].astype(np.int32), axis=1)
    y_media_val = np.argmax(media_data['dev'].astype(np.int32), axis=1)
    y_media_test = np.argmax(media_data['test'].astype(np.int32), axis=1)
    y_emotion_train = np.argmax(emotion_data['train'].astype(np.int32), axis=1)
    y_emotion_val = np.argmax(emotion_data['dev'].astype(np.int32), axis=1)
    y_emotion_test = np.argmax(emotion_data['test'].astype(np.int32), axis=1)
    
    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel
    
    # Print the shapes for potential debugging
    print('Train data shape: ', X_train.shape)
    print('Media train labels shape: ', y_media_train.shape, y_media_train.dtype)
    print('Emotion train labels shape: ', y_emotion_train.shape, y_emotion_train.dtype)
    print('Validation data shape: ', X_val.shape)
    print('Media validation labels shape: ', y_media_val.shape, y_media_val.dtype)
    print('Emotion validation labels shape: ', y_emotion_val.shape, y_emotion_val.dtype)
    print('Test data shape: ', X_test.shape)
    print('Media test labels shape: ', y_media_test.shape, y_media_test.dtype)
    print('Emotion test labels shape: ', y_emotion_test.shape, y_emotion_test.dtype)
    
    # Return tuples of train, dev, test
    train_data = (X_train, y_media_train, y_emotion_train)
    val_data = (X_val, y_media_val, y_emotion_val)
    test_data = (X_test, y_media_test, y_emotion_test)
    
    return train_data, val_data, test_data

class Dataset(object):
    def __init__(self, X, y_media, y_emotion, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y_media, y_emotion: Numpy arrays of media and emotion labels, of any 
        shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y_media.shape[0], 'Got different numbers of data and media labels'
        assert X.shape[0] == y_emotion.shape[0], 'Got different numbers of data and emotin labels'
        self.X, self.y_media, self.y_emotion = X, y_media, y_emotion
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y_media[i:i+B], self.y_emotion[i:i+B]) for i in range(0, N, B))