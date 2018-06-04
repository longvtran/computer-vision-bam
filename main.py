#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:24:17 2018

@author: longtran
"""

"""This is the top-level file to train, evaluate or test your model"""

import sys
import time
import os
import numpy as np
from argparse import ArgumentParser
from data_utils import split_data, load_data, DATA_DIR, INPUT_FILE, MEDIA_LABEL_FILE, EMOTION_LABEL_FILE
from preprocessing import load_data, preprocess
from train import train

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="start mode, train, eval, gen_data, test",
                        metavar="MODE", default="train")
    parser.add_argument("--update", dest="update",
                        help="flag to specify that data should be recreated from the source mages or loaded from saved numpy arrays",
                        action='store_true')
    parser.add_argument("--no-update", dest="update",
                        help="flag to specify that data should be recreated from the source mages or loaded from saved numpy arrays",
                        action='store_false')
    parser.set_defaults(update=False)
    parser.add_argument("--remove_broken", dest="remove_broken",
                        help="flag to specify if images that cannot be loaded from disk should be removed (some BAM images are corrupt)",
                        action='store_true')
    parser.add_argument("--no-remove_broken", dest="remove_broken",
                        help="flag to specify if images that cannot be loaded from disk should be removed (some BAM images are corrupt)",
                        action='store_false')
    parser.set_defaults(remove_broken=False)
    # Augment option defaults to False
    parser.add_argument("--augment", dest="augment",
                        help="flag to specify if images should be augmented)",
                        action='store_true')
    parser.add_argument("--no-augment", dest="augment",
                        help="flag to specify if images should be augmented)",
                        action='store_false')
    parser.set_defaults(augment_data=False)
    parser.add_argument("--device", dest="device", default="cpu",
                        help="device to be used to train")
    parser.add_argument("--log_folder", dest="log_folder",
                        help="log folder to save log files from training")
    return parser

def main():
    """
    Wrapper to run the classification task
    """
    # Parse command-line arguments
    parser = build_parser()
    options = parser.parse_args()
    
    if options.mode == "gen_data":
        # Split the data into train/dev/test sets
        split_data()

        # Load the data and reshape for training and evaluation
        X, y_media, y_emotion = load_data(update=options.update, 
                                          remove_broken=options.remove_broken)
    
        for set_type in ["train", "dev", "test"]:
            total_media = np.sum(y_media[set_type], axis=0)
            total_emotion = np.sum(y_emotion[set_type], axis=0)
    
            print(f"Total images for each media category in {set_type} set:")
            for v, k in enumerate(MEDIA_LABELS):
                print(f"\t{k}: {total_media[v]}")
            print(f"Total images for each emotion category in {set_type} set:")
            for v, k in enumerate(EMOTION_LABELS):
                print(f"\t{k}: {total_emotion[v]}")

    
    elif options.mode == "train":
        # Create directory to save the results
        results_dir = "results"
        if not os.path.exists("./" + results_dir):
            os.makedirs("./" + results_dir)
        # Check if the given log folder already exists
        results_subdirs = os.listdir("./" + results_dir)
        if not options.log_folder:
            raise Exception('Please specify log_folder argument to store results.')
        elif options.log_folder in results_subdirs:
            raise Exception('The given log folder already exists.')
        else:
            # Create a folder for each training run
            log_folder = os.path.join(results_dir, options.log_folder)
            os.makedirs(log_folder)
        
        # Load the data and organize into three tuples (train, val/dev, test)
        # Each tuple consists of input arrays, media labels, and emotion labels
        train_data, val_data, test_data = load_data(DATA_DIR, INPUT_FILE, 
                                                     MEDIA_LABEL_FILE, EMOTION_LABEL_FILE)
        
        # Preprocess the data
        train_dset, val_dset, test_dset = preprocess(train_data, val_data, test_data, 
                                                     augment=options.augment)
        
        # Specify the device:
        if options.device == "cpu":
            device = "/cpu:0"
        elif options.device == "gpu":
            device = "/device:GPU:0"
        
        # Train the model
        train(train_dset, val_dset, log_folder=log_folder, device=device, 
              batch_size=64, num_epochs=100)
        
    elif options.mode == "train_vgg":
        # Load the VGG19 model, add a few layers on top
        pass
    
    elif options.mode == "test":
        # TO BE IMPLEMENTED
        pass

if __name__ == "__main__":
    main()