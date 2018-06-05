#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 22:05:32 2018

@author: longtran
"""

import numpy as np
import tensorflow as tf
import sklearn.metrics
import sys

def convert_to_one_hot(preds):
    class_preds = np.argmax(preds, axis=1)
    return np.eye(preds.shape[1])[class_preds.reshape(-1)]

def evaluate_test(model_path, model_type, test_dset, batch_size=64, confusion_mat=False):
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
        print(model.metrics_names[i])
        print(results[i])
    
    if confusion_mat:
        y_media_pred, y_emotion_pred = model.predict(x_test, batch_size=batch_size)
        y_media_test_label = np.argmax(y_media_test, axis=1)
        y_emotion_test_label = np.argmax(y_emotion_test, axis=1)
        y_media_pred_label = np.argmax(y_media_pred, axis=1)
        y_emotion_pred_label = np.argmax(y_emotion_pred, axis=1)
        
        
        
        cm_media = sklearn.metrics.confusion_matrix(y_media_test_label, y_media_pred_label)
        cm_emotion = sklearn.metrics.confusion_matrix(y_emotion_test_label, y_emotion_pred_label)
        print("Confusion matrix for media:")
        print(cm_media)
        print("Confusion matrix for emotion:")
        print(cm_emotion)
    