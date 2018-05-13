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
import tensorflow as tf
import numpy as np
import util
from tensorflow.python import debug as tf_debug
from argparse import ArgumentParser
from data_utils import load_data

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="start mode, train, eval, gen_data, test",
                        metavar="MODE", default="train")
    parser.add_argument("--update", dest="update",
                        help="flag to specify if data should be recreated from the source mages or loaded from saved numpy arrays",
                        default="false")
    parser.add_argument("remove_broken", dest="remove_broken",
                        help="flag to specify if images that cannot be loaded from disk should be removed (some BAM images are corrupt)",
                        default="false")
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
        X, y_media, y_emotion = load_data(update=options.mode, 
                                          remove_broken=options.remove_broken)
        total_media = np.sum(y_media, axis=0)
        total_emotion = np.sum(y_emotion, axis=0)

        print("Total images for each media category:")
        for v, k in enumerate(MEDIA_LABELS):
            print(f"\t{k}: {total_media[v]}")
        print("Total images for each emotion category:")
        for v, k in enumerate(EMOTION_LABELS):
            print(f"\t{k}: {total_emotion[v]}")
    
    elif options.mode == "train":
        # TO BE IMPLEMENTED
        pass
    
    elif options.mode == "eval":
        # TO BE IMPLEMENTED
        pass
    
    elif options.mode == "test":
        # TO BE IMPLEMENTED
        pass