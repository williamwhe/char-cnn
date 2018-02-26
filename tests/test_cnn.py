import os
import string

import numpy as np
import pandas as pd

from charcnn import cnn
from charcnn import data


class TestModel:
    "Keras model"

    def test_constructs_and_compiles_char_cnn(self):
        n_vocab, max_len, n_classes = 83, 453, 12
        model = cnn.compiled(cnn.char_cnn(n_vocab, max_len, n_classes))
        assert model.built, "model not built"

    def test_builds_estimator(self):
        n_vocab, max_len, n_classes = 83, 453, 12
        model = cnn.compiled(cnn.char_cnn(n_vocab, max_len, n_classes))
        estimator = cnn.estimator(model)
        assert len(estimator.get_variable_names()) > 0

    def test_training_completes(self):
        vocab = list(string.printable)
        classes = ['hi', 'bye', 'unk']
        max_len = 5000

        # preprocessed data
        xtrain = lines('data/test/xtrain.txt')
        xtrain = data.encode_features(xtrain, vocab, max_len=max_len)
        ytrain = lines('data/test/ytrain.txt')
        ytrain = data.encode_labels(ytrain, classes)
        xtest = lines('data/test/xtest.txt')
        xtest = data.encode_features(xtest, vocab, max_len=max_len)

        # train
        model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, len(classes)))
        estimator = cnn.estimator(model)
        cnn.train(estimator, xtrain, ytrain)

        # check results. don't expect it to have learned anything meaningful
        # on 5 instances.
        predictions = cnn.predict(estimator, xtest)
        for p in predictions:
            assert p[0] >= 0.0
            assert p[0] <= 1.0
            assert p[1] >= 0
            assert p[1] <= 1


# Testing utilty functions

def lines(filename):
    with open(filename) as f:
        return f.read().splitlines()
