#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:31:23 2018

@author: longtran
"""

import numpy as np
import tensorflow as tf
import os
from model2 import ConvNet
from preprocessing import gen_generator

def train(train_dset, val_dset, train_datagen, log_folder, device='/cpu:0', 
          batch_size=64, num_epochs=1):
    x, y_media, y_emotion = train_dset
    x_val, y_media_val, y_emotion_val = val_dset
    model = ConvNet()
    
    # summarize layers
    # print(model.summary())
    
    # Compile the model
    optimizer = tf.keras.optimizers.Nadam(lr=0.002)
    model.compile(optimizer=optimizer, 
                  loss={'output_media': 'categorical_crossentropy', 
                        'output_emotion': 'categorical_crossentropy'},
                  loss_weights={'output_media': 1., 'output_emotion': 1},
                  metrics=['accuracy'])
    
    # Save training results to a log file
    log_file = os.path.join(log_folder, 'training_log.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(log_file)
    
    # Create tensorboard 
    tsb_dir = os.path.join(log_folder, 'tensorboard')
    tsb_logger = tf.keras.callbacks.TensorBoard(log_dir=tsb_dir,
                                                histogram_freq=5,
                                                batch_size=batch_size, 
                                                write_graph=False, 
                                                write_grads=False,
                                                write_images=True)
    
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
    model.fit_generator(gen_generator(train_datagen, x, y_media, y_emotion, batch_size=batch_size),
               epochs=num_epochs, steps_per_epoch= np.ceil(len(x) / batch_size),
               validation_data=(x_val, [y_media_val, y_emotion_val]),
               callbacks=[csv_logger, tsb_logger, media_ckpt, emotion_ckpt])
    
    # Save the final model
    model_file = os.path.join(log_folder, 'last_ckpt.h5')
    model.save(model_file)
    
    