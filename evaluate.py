#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 22:05:32 2018

@author: longtran
"""

import tensorflow as tf

def evaluate_test(model_path, model_type, test_dset, batch_size=64):
    x_test, y_media_test, y_emotion_test = test_dset
    
    if model_type == "mobile":
#        model = tf.keras.models.load_model(model_path, 
#                                           custom_objects={'relu6': tf.keras.applications.mobilenet.relu6,
#                                                           'DepthwiseConv2D': tf.keras.applications.mobilenet.DepthwiseConv2D})
        from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
        from tensorflow.python.keras._impl.keras.applications import mobilenet
        from tensorflow.python.keras._impl.keras.models import load_model
        with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
            model = load_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)
    
    results = model.evaluate(x_test, {'output_media': y_media_test, 'output_emotion': y_emotion_test},
                   batch_size=batch_size, verbose=True)
    for i in range(0, len(results)):
        print(model.metrics_names[i] + ": " + results[i])
    