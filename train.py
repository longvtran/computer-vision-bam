#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:31:23 2018

@author: longtran
"""

import numpy as np
import tensorflow as tf
import os
from model6 import ConvNet



def train(train_dset, val_dset, log_folder, device='/cpu:0', batch_size=64, num_epochs=1,
          model_type="custom"):
    x, y_media, y_emotion = train_dset
    x_val, y_media_val, y_emotion_val = val_dset
    if model_type == "custom":
        print("Using custom model...")
        model = ConvNet()
    elif model_type == "vgg19":
        from model_vgg19 import VGG19
        print("Using pre-trained VGG19 model...")
        model = VGG19()
    elif model_type == "vgg16":
        from model_vgg16 import VGG16
        print("Using pre-trained VGG16 model...")
        model = VGG16()
    elif model_type == "mobile":
        from model_mobile import MobileNet
        print("Using pre-trained MobileNet model...")
        model = MobileNet()
    elif model_type == "nasnet":
        from model_nasnet import NASNet
        print("Using pre-trained NASNetMobile model...")
        model = NASNet()
#    elif model_type == "densenet":
#        from model_densenet import DenseNet
#        print("Using pre-trained DenseNetMobile model...")
#        model=DenseNet()
    elif model_type == "xception":
        from model_xception import Xception
        print("Using pre-trained Xception model...")
        model = Xception()
        
    # summarize layers
    # print(model.summary())
    
    # Compile the model
    optimizer = tf.keras.optimizers.Nadam(lr=0.002)
    model.compile(optimizer=optimizer, 
                  loss={'output_media': 'categorical_crossentropy', 
                        'output_emotion': 'categorical_crossentropy'},
                  loss_weights={'output_media': 1, 'output_emotion': 1.},
                  metrics=['accuracy'])
    
    # Save training results to a log file
    log_file = os.path.join(log_folder, 'training_log.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(log_file)
    
    # Create tensorboard 
    tsb_dir = os.path.join(log_folder, 'tensorboard')
    tsb_logger = tf.keras.callbacks.TensorBoard(log_dir=tsb_dir,
                                                histogram_freq=0,
                                                batch_size=batch_size, 
                                                write_graph=False, 
                                                write_grads=False,
                                                write_images=False)
    
    # Store best models during training
    media_ckpt_file = os.path.join(log_folder, 'media_ckpt_best.h5')
    media_ckpt = tf.keras.callbacks.ModelCheckpoint(media_ckpt_file, 
                                              monitor='val_output_media_acc',
                                              verbose=1, save_weights_only=False,
                                              save_best_only=True, mode='auto',
                                              period=1)
    
    emotion_ckpt_file = os.path.join(log_folder, 'emotion_ckpt_best.h5')
    emotion_ckpt = tf.keras.callbacks.ModelCheckpoint(emotion_ckpt_file, 
                                              monitor='val_output_emotion_acc',
                                              verbose=1, save_weights_only=False,
                                              save_best_only=True, mode='auto',
                                              period=1)    
    
    # Train the model
    model.fit(x, {'output_media': y_media, 'output_emotion': y_emotion},
               epochs=num_epochs, batch_size=batch_size,
               validation_data=(x_val, [y_media_val, y_emotion_val]),
               callbacks=[csv_logger, tsb_logger, media_ckpt, emotion_ckpt])
    
    # Save the final model
    model_file = os.path.join(log_folder, 'last_ckpt.h5')
    model.save(model_file)
    
    