#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 03:36:48 2018

@author: longtran
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:38:44 2018

@author: longtran
"""

import tensorflow as tf

def NASNet(num_classes_media=7, num_classes_emotion=4, training=False):
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
    
    # get the base NASNet model
    base_nasnet = tf.keras.applications.NASNetMobile(include_top=False,
                                                     weights='imagenet',
                                                   input_tensor=inputs,
                                                   input_shape=(128,128,3))
    
    # Freeze all except the last five layers of NASNet (4 conv and 1 pooling layer)
    for layer in base_nasnet.layers[:-5]:
        layer.trainable = False
    
    # Add our layers on top
    last_x = base_nasnet.layers[-1].output
    
    # Media side
    x_media = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', 
                               activation=tf.nn.relu, 
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-3))(last_x)
    x_media = tf.keras.layers.MaxPooling2D(2, 2)(x_media)
    x_media = tf.keras.layers.BatchNormalization()(x_media)
    x_media = tf.keras.layers.Flatten()(x_media)
    x_media = tf.keras.layers.Dense(units=1024, 
                                    kernel_regularizer=tf.keras.regularizers.l2(5e-3),
                                    activation=tf.nn.relu)(x_media)
    x_media = tf.keras.layers.Dropout(rate=0.6)(x_media)
    output_media = tf.keras.layers.Dense(num_classes_media,
                                   kernel_initializer=initializer,
                                   activation='softmax',
                                   name='output_media')(x_media)
    
    # Emotion side
    x_emotion = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', 
                                 activation=tf.nn.relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3))(last_x)
    x_emotion = tf.keras.layers.MaxPooling2D(2, 2)(x_emotion)
    x_emotion = tf.keras.layers.BatchNormalization()(x_emotion)
    x_emotion = tf.keras.layers.Flatten()(x_emotion)
    x_emotion = tf.keras.layers.Dense(units=1024, 
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      activation=tf.nn.relu)(x_emotion)
    x_emotion = tf.keras.layers.Dropout(rate=0.6)(x_emotion)
    output_emotion = tf.keras.layers.Dense(num_classes_emotion,
                                     kernel_initializer=initializer,
                                     activation='softmax',
                                     name='output_emotion')(x_emotion)
    
    # Return the complete model
    model = tf.keras.Model(inputs=inputs, outputs=[output_media, output_emotion])
    
    return model