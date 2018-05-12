#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:38:42 2018

@author: longtran
"""

import sqlite3
import urllib
import pandas as pd
import numpy as np

#DATA_DIR = "20170509-bam-1m-18UThu3ICM.sqlite"
DATA_DIR = "20170509-bam-crowd-only-xQ3gXol5UR.sqlite"

# Save scores table as a pkl file

scores = pd.read_sql("select * from scores",
                 sqlite3.connect(DATA_DIR))
scores['mid'] = scores['mid'].apply(np.int64)

modules = pd.read_sql("select * from modules", sqlite3.connect(DATA_DIR))
modules['mid'] = modules['mid'].apply(np.int64)
#join_df = pd.concat([scores, modules], axis=1, join='inner', join_axis=scores.mid)

#join_df = pd.merge(scores, modules, how='inner', on='mid')
#join_df = join_df.sort(['mid'])
#print(join_df['mid'])


#crowd = pd.read_sql("select * from crowd_labels where attribute='emotion_happy'",
#                 sqlite3.connect(DATA_DIR))
#print(crowd.iloc[0])

auto_labels = pd.read_sql("select * from automatic_labels",
                 sqlite3.connect(DATA_DIR))
print(auto_labels.iloc[0])
join_df = pd.merge(auto_labels, modules, how='inner', on='mid')

head_df = join_df.iloc[0:3000]

#print(head_df.describe())

#print(head_df[head_df.media_comic=="positive"].count())

print(head_df['media_comic'].value_counts())
print(head_df['media_3d_graphics'].value_counts())
print(head_df['media_graphite'].value_counts())
print(head_df['media_vectorart'].value_counts())
print(head_df['media_oilpaint'].value_counts())
print(head_df['media_pen_ink'].value_counts())
print(head_df['media_watercolor'].value_counts())