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
from data_utils import split_data, load_data, load_image, DATA_DIR, TRAIN_STATS_DIR, INPUT_FILE, MEDIA_LABEL_FILE, EMOTION_LABEL_FILE
from preprocessing import load_data, preprocess, preprocess_from_file, preprocess_image
from train import train
from evaluate import evaluate_test, evaluate_ensemble, predict_image


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
    parser.set_defaults(augment=False)
    # Model type
    parser.add_argument("--model_type", dest="model_type",
                        help="flag to specify which model to train/test: custom, vgg19, vgg16, mobile, xception",
                        default="custom")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="device to be used to train")
    parser.add_argument("--log_folder", dest="log_folder",
                        help="log folder to save log files from training")
    # Model name to test (copy the models to be tested to the test_models folder)
    parser.add_argument("--model_name", dest="model_name",
                        help="flag to specify the name of the model to test")
    # Confusion matrix option defaults to False
    parser.add_argument("--confusion_mat", dest="confusion_mat",
                        help="flag to specify if images should be augmented)",
                        action='store_true')
    parser.add_argument("--no-confusion_mat", dest="confusion_mat",
                        help="flag to specify if images should be augmented)",
                        action='store_false')
    parser.set_defaults(confusion_mat=False)
    # Where to find the ensemble folder that stores test models
    parser.add_argument("--ensemble_folder", dest="ensemble_folder",
                        help="ensemble folder where models in the ensemble are stored")
    # Image name to test
    parser.add_argument("--image", dest="image",
                        help="flag to specify the name of the image to test")
    
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
                                                     augment=options.augment, 
                                                     train_stats_dir=TRAIN_STATS_DIR)
        
        # Specify the device:
        if options.device == "cpu":
            device = "/cpu:0"
        elif options.device == "gpu":
            device = "/device:GPU:0"
        
        # Train the model
        train(train_dset, val_dset, log_folder=log_folder, device=device, 
              batch_size=64, num_epochs=100, model_type=options.model_type)
    
    elif options.mode == "test":
        # Load the data and organize into three tuples (train, val/dev, test)
        # Each tuple consists of input arrays, media labels, and emotion labels
        train_data, val_data, test_data = load_data(DATA_DIR, INPUT_FILE, 
                                                     MEDIA_LABEL_FILE, EMOTION_LABEL_FILE)
        # Preprocess the data
        if os.path.isfile(os.path.join(TRAIN_STATS_DIR, "train_stats.npz")):
            print("Preprocess test data using saved statistics from train data...")
            train_stats_file = os.path.join(TRAIN_STATS_DIR, "train_stats.npz")
            test_dset = preprocess_from_file(train_stats_file, test_data, augment=options.augment)
        else:
            print("Preprocess test data using train data...")
            train_dset, val_dset, test_dset = preprocess(train_data, val_data, test_data, 
                                                         augment=options.augment,
                                                         train_stats_dir=TRAIN_STATS_DIR)

        # Specify the device:
        if options.device == "cpu":
            device = "/cpu:0"
        elif options.device == "gpu":
            device = "/device:GPU:0"
            
        # Load the model
        model_path = os.path.join("test_models", options.model_name)
        evaluate_test(model_path, options.model_type, test_dset, batch_size=64, 
                      confusion_mat=options.confusion_mat)
    
    elif options.mode == "ensemble":
        # Load the data and organize into three tuples (train, val/dev, test)
        # Each tuple consists of input arrays, media labels, and emotion labels
        train_data, val_data, test_data = load_data(DATA_DIR, INPUT_FILE, 
                                                     MEDIA_LABEL_FILE, EMOTION_LABEL_FILE)
        # Preprocess the data
        if os.path.isfile(os.path.join(TRAIN_STATS_DIR, "train_stats.npz")):
            print("Preprocess test data using saved statistics from train data...")
            train_stats_file = os.path.join(TRAIN_STATS_DIR, "train_stats.npz")
            test_dset = preprocess_from_file(train_stats_file, test_data, augment=options.augment)
        else:
            print("Preprocess test data using train data...")
            train_dset, val_dset, test_dset = preprocess(train_data, val_data, test_data, 
                                                         augment=options.augment,
                                                         train_stats_dir=TRAIN_STATS_DIR)
        # Specify the device:
        if options.device == "cpu":
            device = "/cpu:0"
        elif options.device == "gpu":
            device = "/device:GPU:0"
        
        if not options.ensemble_folder:
            raise Exception('Please specify ensemble_folder argument to find ensemble folders.')
        elif len(os.listdir(options.ensemble_folder)) == 0:
            raise Exception('Ensemble folder is empty.')
        
        # Evaluate the ensemble
        evaluate_ensemble(options.ensemble_folder, test_dset, batch_size=64, 
                          confusion_mat=options.confusion_mat) 
    
    elif options.mode == "test_single":
        x_test = load_image(os.path.join('stylized_images_configs', options.image))
        train_stats_file = os.path.join(TRAIN_STATS_DIR, "train_stats.npz")
        x_test = preprocess_image(train_stats_file, x_test, augment=options.augment)
        
        model_path = os.path.join("test_models", options.model_name)
        predict_image(x_test, model_path)


if __name__ == "__main__":
    main()
