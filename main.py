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
from data_utils import split_data, load_data, DATA_DIR, INPUT_FILE, MEDIA_LABELS, EMOTION_LABELS
from preprocessing import preprocess, Dataset
from train import train, model_init_fn, optimizer_init_fn

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
    parser.add_argument("--processes", dest="processes",
                        help="number of processes you want to start to train the network",
                        default="1")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="device to be used to train")
    parser.add_argument("--folder", dest="folder", type=int,
                        help="folder(int) to load the config, neglect this option if loading from ./computer-vision-bam/net_config.json")
    return parser

def main():
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
        train_data, val_data, test_data = preprocess(DATA_DIR, INPUT_FILE, MEDIA_LABELS, EMOTION_LABELS)
        train_dset = Dataset(train_data[0], train_data[1], train_data[2], batch_size=64, shuffle=True)
        val_dset = Dataset(val_data[0], val_data[1], val_data[2], batch_size=64, shuffle=False)
        train(model_init_fn, optimizer_init_fn, train_dset, val_dset, num_epochs=20)
        
    elif options.mode == "eval":
        # TO BE IMPLEMENTED
        pass
    
    elif options.mode == "test":
        # TO BE IMPLEMENTED
        pass

if __name__ == "__main__":
    main()