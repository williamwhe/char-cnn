#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An implementation of

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)

"""

import numpy as np
import json

import tensorflow as tf

from charcnn import data


def build(vocab, max_len, classes):
    """
    Build estimator
    """

    ret = char_cnn(len(vocab), max_len, len(classes))
    ret = compiled(ret)
    ret = estimator(ret)

    return ret


def char_cnn(n_vocab, max_len, n_classes):
    """
    See Zhang and LeCun, 2015.
    """

    # reserve for special control characters – eg unknown, padding
    n_vocab = n_vocab + data.N_VOCAB_RESERVED

    # inputs
    inputs = tf.keras.Input(shape=(max_len, n_vocab), name='chars_input')

    x = tf.keras.layers.Conv1D(256, 7, activation='relu')(inputs)
    x = tf.keras.layers.MaxPool1D(3)(x)

    x = tf.keras.layers.Conv1D(256, 7, activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(3)(x)

    x = tf.keras.layers.Conv1D(256, 3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool1D(3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # output
    predictions = tf.keras.layers.Dense(n_classes, activation='softmax', name='labels')(x)

    # construct model
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model


def compiled(model):
    "compile with chosen config"

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def estimator(model, model_dir=None):
    "build tensorflow estimator"

    return tf.keras.estimator.model_to_estimator(keras_model=model,
                                                 model_dir=model_dir)


def predict(estimator, input_fn):
    "predict probability, class for each instance"

    # predict probability of each class for each instance
    all_preds = np.array([y['labels']
                          for y
                          in estimator.predict(input_fn)])

    # for each instance get the index of the class with max probability
    idxs = np.argmax(all_preds, axis=1)

    # get the values of the highest probability for each instance
    preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]

    return np.array(preds), idxs
