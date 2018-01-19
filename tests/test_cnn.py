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

    def test_training(self):
        xtrain, ytrain, xtest, vocab, max_len, n_classes = data.preprocess(
            lines('data/test/xtrain.txt'),
            lines('data/test/ytrain.txt'),
            lines('data/test/xtest.txt'),
            max_len=130)

        model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, n_classes))
        history = cnn.fit(model, xtrain, ytrain)


# Testing utilty functions

def lines(filename):
    with open(filename) as f:
        return f.read().splitlines()
