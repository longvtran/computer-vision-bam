"""
@author: Abhishek
"""

from __future__ import print_function
#import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

ch1=32
ch2=64
fsz=3
pool=2
dense1=512

batch_size= 32
num_classes=11
epochs=50
learning_rate= 0.001
dr= 1e-6
data_augmentation=True
num_predictions=22

def weight_initialization(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev= 0.05))

def bias_initialization(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def my_model(x_train):
    model=Sequential()
    model.add(Conv2D(ch1,(fsz,fsz), padding='same',input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(ch1,(fsz,fsz)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool,pool)))
    model.add(Dropout(0.25))

    model.add(Conv2D(ch2, (fsz,fsz), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(ch2,(fsz,fsz)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool,pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(dense1))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def optimizer(learning_rate, dr):
    optimizer= keras.optimizers.rmsprop(lr=learning_rate, decay= dr)
    return optimizer

def my_model_compile(model, opt):
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def normalize_ip(x_train, x_test):
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, x_test)

def data_aug_train(model, data_augmentation, x_train, y_train, batch_size,epochs):
    if not data_augmentation:
        print('Without data augmentation:')
        model.fit(x_train, y_train, batch_size= batch_size,
                epochs=epochs,validation_date=(x_test, y_test), shuffle=True)

    else:
        print('Real time data augmentation:')
        datagen= ImageDataGenerator(featurewise_center= False,
                samplewise_center= False, featurewise_std_normalization= False,
                samplewise_std_normalization= False, zca_whitening= False,
                rotation_range=0, width_shift_range= 0.1, height_shift_range=
                0.1, horizontal_flip= True, vertical_flip=False)
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs= epochs, validation_data= (x_test, y_test), workers=4)
    return model

def save_model(model, save_dir, model_name):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_pwpath= os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Train model saved at %s' %model_path)
    return model

def model_evaluate(model, x_test, y_test):
    a= model.evaluate(x_test, y_test, verbose= 1)
    print('Test loss :', a[0])
    print('Test_accuracy:', a[1])


save_dir= os.path.join(os.getcwd(), 'saved_models')
model_name= 'keras_cifar10_trained_model.h5'

#Data split
(x_train, y_train), (x_test,y_test)= cifar10.load_data()
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

y_train=keras.utils.to_categorical(y_train, num_classes)
y_test= keras.utils.to_categorical(y_test, num_classes)

model= my_model(x_train)
opt= optimizer(learning_rate, dr)

model=  my_model_compile(model, opt)
x_train, x_test= normalize_ip(x_train, x_test)
model= data_aug_train(model, data_augmentation, x_train, y_train, batch_size,epochs)

save_model(model, save_dir, model_name)
model_evaluate(model, x_test, y_test)
