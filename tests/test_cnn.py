import os

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
        xtrain, ytrain, xtest, vocab, max_len, n_classes = data.preprocess(
            lines('data/test/xtrain.txt'),
            lines('data/test/ytrain.txt'),
            lines('data/test/xtest.txt'),
            max_len=130)

        model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, n_classes))
        history = cnn.fit(model, xtrain, ytrain)
        probabilities, classes = cnn.predict(model, xtest)

        # don't expect it to have learned anything meaningful on 5 instances
        for p in probabilities:
            assert p >= 0.0
            assert p <= 1.0

        for c in classes:
            assert c >= 0
            assert c <= 1


# Testing utilty functions

def lines(filename):
    with open(filename) as f:
        return f.read().splitlines()
