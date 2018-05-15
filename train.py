#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:31:23 2018

@author: longtran
"""

import tensorflow as tf
from simple_model import ConvNet

def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.
    
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
      
    Returns: Nothing, but prints the accuracy of the model
    """
    num_media_correct = 0
    num_emotion_correct = 0
    num_samples = 0
    for x_batch, y_media_batch, y_emotion_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        #scores_media_np, scores_emotion_np = sess.run([scores_media, scores_emotion], feed_dict=feed_dict)
        scores_np = sess.run(scores, feed_dict=feed_dict)
        scores_media_np = scores_np[:,:7]
        scores_emotion_np = scores_np[:,7:]
        y_media_pred = scores_media_np.argmax(axis=1)
        y_emotion_pred = scores_emotion_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_media_correct += (y_media_pred == y_media_batch).sum()
        num_emotion_correct += (y_emotion_pred == y_emotion_batch).sum()
    media_acc = float(num_media_correct) / num_samples
    emotion_acc = float(num_emotion_correct) / num_samples
    print('Got %d / %d correct (%.2f%%) for media labels' % (num_media_correct, num_samples, 100 * media_acc))
    print('Got %d / %d correct (%.2f%%) for emotion labels' % (num_emotion_correct, num_samples, 100 * emotion_acc))

def model_init_fn(inputs, is_training):
    model = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    model = ConvNet()
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return model(inputs)

def optimizer_init_fn(learning_rate=3e-3):
    optimizer = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer

def train(model_init_fn, optimizer_init_fn, train_dset, val_dset, device='/cpu:0',
          num_epochs=1, print_every=30):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.
    
    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for
    
    Returns: Nothing, but prints progress during trainingn
    """
    tf.reset_default_graph()    
    with tf.device(device):
        # Construct the computational graph we will use to train the model. We
        # use the model_init_fn to construct the model, declare placeholders for
        # the data and labels
        x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        y_media = tf.placeholder(tf.int32, [None])
        y_emotion = tf.placeholder(tf.int32, [None])
        
        # We need a place holder to explicitly specify if the model is in the training
        # phase or not. This is because a number of layers behaves differently in
        # training and in testing, e.g., dropout and batch normalization.
        # We pass this variable to the computation graph through feed_dict as shown below.
        is_training = tf.placeholder(tf.bool, name='is_training')
                
        # Use the model function to build the forward pass.
        #scores_media, scores_emotion = model_init_fn(x, is_training)
        scores = model_init_fn(x, is_training)
        
        # Compute the losses
        scores_media = scores[:,:7]
        scores_emotion = scores[:,7:]
        loss_media = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_media, logits=scores_media)
        loss_emotion = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_emotion, logits=scores_emotion)
        loss = tf.reduce_mean(loss_media + loss_emotion)
        
        # Use the optimizer_fn to construct an Optimizer, then use the optimizer
        # to set up the training step. Asking TensorFlow to evaluate the
        # train_op returned by optimizer.minimize(loss) will cause us to make a
        # single update step using the current minibatch of data.
        
        # Note that we use tf.control_dependencies to force the model to run
        # the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
        # holds the operators that update the states of the network.
        # For example, the tf.layers.batch_normalization function adds the running mean
        # and variance update operators to tf.GraphKeys.UPDATE_OPS.
        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    # Now we can run the computational graph many times to train the model.
    # When we call sess.run we ask it to evaluate train_op, which causes the
    # model to update.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            for x_np, y_media_np, y_emotion_np in train_dset:
                feed_dict = {x: x_np, y_media: y_media_np, y_emotion: y_emotion_np, is_training:1}
                #feed_dict = {x: x_np, y_media: y_media_np, is_training:1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
                    print()
                t += 1
            
            print("Validation Accuracy:")
            check_accuracy(sess, val_dset, x, scores, is_training=is_training)
            print("Training Accuracy:")
            check_accuracy(sess, train_dset, x, scores, is_training=is_training)
            print()