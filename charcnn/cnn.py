#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An implementation of

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)

"""

import numpy as np
import json

from tensorflow.python import keras as ks
import tensorflow as tf

from charcnn import data


def char_cnn(n_vocab, max_len, n_classes, weights_path=None):
    "See Zhang and LeCun, 2015"

    model = ks.models.Sequential()
    model.add(ks.layers.Conv1D(256, 7, activation='relu', input_shape=(max_len, n_vocab), name='chars'))
    model.add(ks.layers.MaxPool1D(3))

    model.add(ks.layers.Conv1D(256, 7, activation='relu'))
    model.add(ks.layers.MaxPool1D(3))

    model.add(ks.layers.Conv1D(256, 3, activation='relu'))
    model.add(ks.layers.Conv1D(256, 3, activation='relu'))
    model.add(ks.layers.Conv1D(256, 3, activation='relu'))
    model.add(ks.layers.Conv1D(256, 3, activation='relu'))
    model.add(ks.layers.MaxPool1D(3))

    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(1024, activation='relu'))
    model.add(ks.layers.Dropout(0.5))
    model.add(ks.layers.Dense(1024, activation='relu'))
    model.add(ks.layers.Dropout(0.5))
    model.add(ks.layers.Dense(n_classes, activation='softmax', name='labels'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def compiled(model):
    "compile with chosen config"

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def estimator(model):
    "build tensorflow estimator"

    return ks.estimator.model_to_estimator(keras_model=model)


def input_function(features,
                   labels=None,
                   shuffle=False,
                   batch_size=128,
                   num_epochs=5):
    """
    returns estimator input function
    """

    # estimators expects flaot32
    features = features.astype(np.float32)
    if labels is not None:
        labels = labels.astype(np.float32)

    # return function () -> (features, labels)
    return tf.estimator.inputs.numpy_input_fn(
        x={'chars_input': features},
        y=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=num_epochs)


def train(estimator,
          xtrain,
          ytrain,
          batch_size=128,
          num_epochs=5):
    """
    train the estimator
    """

    return estimator.train(input_function(xtrain,
                                          ytrain,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_epochs=num_epochs))


def evaluate(estimator,
             xtest,
             ytest):
    """
    evaluate the model
    """

    return estimator.evaluate(input_function(xtest, labels=ytest))


def predict(estimator, X):
    "predict probability, class for each instance"

    # predict probability of each class for each instance
    all_preds = np.array([y['labels']
                          for y
                          in estimator.predict(input_function(X))])

    # for each instance get the index of the class with max probability
    idxs = np.argmax(all_preds, axis=1)

    # get the values of the highest probability for each instance
    preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]

    return np.array(preds), idxs


def main():
    "learn and predict"

    def lines(filename):
        with open(filename) as f:
            return f.read().splitlines()

    # configure
    vocab = list(string.printable)
    classes = data.dbedia_classes()
    max_len = 1014

    # read data
    xtrain = lines('data/test/xtrain.txt')
    ytrain = lines('data/test/ytrain.txt')
    xtest = lines('data/test/xtest.txt')

    # preprocess data
    xtrain = data.encode_features(xtrain, vocab, max_len=max_len)
    ytrain = data.encode_labels(ytrain, classes)
    xtest = data.encode_features(xtest, vocab, max_len=max_len)

    # train
    model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, len(classes)))
    estimator = cnn.estimator(model)
    history = cnn.train(estimator, xtrain, ytrain)
    model.save_weights('weights.h5')

    # write training metrics
    print(history.history)
    with open('metrics.txt', 'w') as f:
        f.write(json.dumps(history.history, indent=1))

    # prediction
    _, ytest = cnn.predict(estimator, xtest)
    with open('ytest.txt', 'w') as f:
        f.write('\n'.join(map(str, ytest)))

    # test set predictions for inspection
    _, ytrain_predicted = cnn.predict(estimator, xtrain)
    with open('ytrain.predicted.txt', 'w') as f:
        f.write('\n'.join(map(str, ytrain_predicted)))


if __name__ == "__main__":
    main()
