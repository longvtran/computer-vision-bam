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

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="start mode, train, eval, download_data", "test",
                        metavar="MODE", default="train")
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
    
    if options.mode == "download_data":
        # TO BE IMPLEMENTED
        pass
    
    elif options.mode == "train":
        # TO BE IMPLEMENTED
        pass
    
    elif options.mode == "eval":
        # TO BE IMPLEMENTED
        pass
    elif options.mode == "test":
        # TO BE IMPLEMENTED
        pass