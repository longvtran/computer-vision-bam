#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:26:33 2018

@author: longtran
"""

import os
import numpy as np
import tensorflow as tf

class ConvNet(tf.keras.Model):
    def __init__(self, num_classes_media=7, num_classes_emotion=4):
        super().__init__()
        ########################################################################
        # TODO: Implement the __init__ method for a  ConvNet. You              #
        # should instantiate layer objects to be used in the forward pass.     #
        ########################################################################
        initializer = tf.variance_scaling_initializer(scale=2.0)
        
        # Media side
        self.conv1_media = tf.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.maxpool1_media = tf.layers.MaxPooling2D(2, 2)
        self.conv2_media = tf.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.maxpool2_media = tf.layers.MaxPooling2D(2, 2)
        self.conv3_media = tf.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.flatten_media = tf.layers.Flatten()
        self.fc_media = tf.layers.Dense(num_classes_media,
                                   kernel_initializer=initializer)
        
        # Emotion side
        self.conv1_emotion = tf.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.maxpool1_emotion = tf.layers.MaxPooling2D(2, 2)
        self.conv2_emotion = tf.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.maxpool2_emotion = tf.layers.MaxPooling2D(2, 2)
        self.conv3_emotion = tf.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same',
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.flatten_emotion = tf.layers.Flatten()
        self.fc_emotion = tf.layers.Dense(num_classes_emotion,
                                   kernel_initializer=initializer)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
    def call(self, x, training=None):
        scores_media, scores_emotion = None, None
        ########################################################################
        # TODO: Implement the forward pass for a ConvNet. You                  #
        # should use the layer objects defined in the __init__ method.         #
        ########################################################################
        # Media side
        x_media = self.conv1_media(x)
        x_media = self.maxpool1_media(x_media)
        x_media = self.conv2_media(x_media)
        x_media = self.maxpool2_media(x_media)
        x_media = self.conv3_media(x_media)
        x_media = self.flatten_media(x_media)
        scores_media = self.fc_media(x_media)
        
        # Emotion side
        x_emotion = self.conv1_emotion(x)
        x_emotion = self.maxpool1_emotion(x_emotion)
        x_emotion = self.conv2_emotion(x_emotion)
        x_emotion = self.maxpool2_emotion(x_emotion)
        x_emotion = self.conv3_emotion(x_emotion)
        x_emotion = self.flatten_emotion(x_emotion)
        scores_emotion = self.fc_emotion(x_emotion)
        
        scores = tf.concat([scores_media, scores_emotion], axis=1)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return scores