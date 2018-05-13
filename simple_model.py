#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:26:33 2018

@author: longtran
"""

import os
import numpy as np
import tensorflow as tf

class ThreeLayerConvNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Implement the __init__ method for a three-layer ConvNet. You   #
        # should instantiate layer objects to be used in the forward pass.     #
        ########################################################################
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.conv1 = tf.layers.Conv2D(channel_1, (5,5), padding='valid', 
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.conv2 = tf.layers.Conv2D(channel_2, (3,3), padding='valid', 
                                     activation=tf.nn.relu, kernel_initializer=
                                     initializer)
        self.fc = tf.layers.Dense(num_classes,
                                   kernel_initializer=initializer)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
    def call(self, x, training=None):
        scores = None
        ########################################################################
        # TODO: Implement the forward pass for a three-layer ConvNet. You      #
        # should use the layer objects defined in the __init__ method.         #
        ########################################################################
        x = tf.pad(x, ((0,0), (2,2), (2,2), (0,0)), "CONSTANT")
        x = self.conv1(x)
        x = tf.pad(x, ((0,0), (1,1), (1,1), (0,0)), "CONSTANT")
        x = self.conv2(x)
        x = tf.layers.flatten(x)
        scores = self.fc(x)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return scores