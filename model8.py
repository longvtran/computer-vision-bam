#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:26:33 2018

@author: longtran
"""

import tensorflow as tf

def ConvNet(num_classes_media=7, num_classes_emotion=4, training=False):
    """
    Build a convolutional network using Keras Functional API. This network reads
    the image image arrays as input and produces two prediction outputs on the
    media or emotion class. The architecture consists of two sequences that share
    weights in the first Conv2D and MaxPooling2D layer
    
    Inputs:
        - num_classes_media: the number of media classes
        - num_classes_emotion: the number of emotion classes
        - training: a boolean that indicates whether the model is currently in
        training phase or not
    
    Returns:
        - a keras model instance
    """
    
    initializer = tf.variance_scaling_initializer(scale=2.0)
    
    inputs = tf.keras.layers.Input(shape=(128,128,3))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', 
                                     activation=tf.nn.relu, 
                                     kernel_initializer=initializer,
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
#    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', 
                                     activation=tf.nn.relu, 
                                     kernel_initializer=initializer,
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inputs)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Media side
    x_media = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', 
                               activation=tf.nn.relu, 
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x_media = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                               activation=tf.nn.relu, 
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x_media = tf.keras.layers.MaxPooling2D(2, 2)(x_media)
    x_media = tf.keras.layers.BatchNormalization()(x_media)
    x_media = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                           activation=tf.nn.relu, 
                           kernel_initializer=initializer,
                           kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x_media = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same',
                               activation=tf.nn.relu, 
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x_media)
    x_media = tf.keras.layers.MaxPooling2D(2, 2)(x_media)
    x_media = tf.keras.layers.Flatten()(x_media)
    x_media = tf.keras.layers.Dense(units=1024, 
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                    activation=tf.nn.relu)(x_media)
    x_media = tf.keras.layers.Dropout(rate=0.5)(x_media)
    x_media = tf.keras.layers.Dense(units=1024, 
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                    activation=tf.nn.relu)(x_media)
    x_media = tf.keras.layers.Dropout(rate=0.5)(x_media)
    output_media = tf.keras.layers.Dense(num_classes_media,
                                   kernel_initializer=initializer,
                                   activation='softmax',
                                   name='output_media')(x_media)
    
    # Emotion side
    x_emotion = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', 
                                 activation=tf.nn.relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x_emotion = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                                 activation=tf.nn.relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x_emotion = tf.keras.layers.MaxPooling2D(2, 2)(x_emotion)
    x_emotion = tf.keras.layers.BatchNormalization()(x_emotion)
    x_emotion = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same',
                                 activation=tf.nn.relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x_emotion)
    x_emotion = tf.keras.layers.MaxPooling2D(2, 2)(x_emotion)
    x_emotion = tf.keras.layers.Flatten()(x_emotion)
    x_emotion = tf.keras.layers.Dense(units=1024, 
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      activation=tf.nn.relu)(x_emotion)
    x_emotion = tf.keras.layers.Dropout(rate=0.4)(x_emotion)
    x_emotion = tf.keras.layers.Dense(units=1024, 
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      activation=tf.nn.relu)(x_emotion)
    x_emotion = tf.keras.layers.Dropout(rate=0.4)(x_emotion)
    output_emotion = tf.keras.layers.Dense(num_classes_emotion,
                                     kernel_initializer=initializer,
                                     activation='softmax',
                                     name='output_emotion')(x_emotion)
    
    # Return the complete model
    model = tf.keras.Model(inputs=inputs, outputs=[output_media, output_emotion])
    
    return model