#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:31:23 2018

@author: longtran
"""

import tensorflow as tf
import os
from model2 import ConvNet

def train(train_dset, val_dset, log_folder, device='/cpu:0', batch_size=64, num_epochs=1, print_every=30):
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
                                                histogram_freq=0,
                                                batch_size=batch_size, 
                                                write_graph=True, write_images=True)
    
    # Train the model
    model.fit(x, 
              {'output_media': y_media, 'output_emotion': y_emotion},
               epochs=num_epochs, batch_size=batch_size,
               validation_data=(x_val, [y_media_val, y_emotion_val]),
               callbacks=[csv_logger, tsb_logger])
    
    # Save the model
    model_file = os.path.join(log_folder, 'model.h5')
    model.save(model_file)
    
    