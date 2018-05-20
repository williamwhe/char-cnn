# -*- coding: utf-8 -*-

import os
import string

import numpy as np
import pandas as pd
import tensorflow as tf

from charcnn import cnn
from charcnn import data

TRAIN_CSV = 'data/test/train.csv.gz'


class TestModel:
    "Tensorflow model"

    def setup_method(self):
        tf.reset_default_graph()

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
        vocab = list('ABCDbdeghilmnosy ,')
        classes = range(3)
        max_len = 300
        n_classes, n_vocab = len(classes), len(vocab)

        # train
        model = cnn.compiled(cnn.char_cnn(n_vocab, max_len, n_classes))
        estimator = cnn.estimator(model)
        estimator.train(data.input_function(TRAIN_CSV,
                                            vocab,
                                            classes,
                                            batch_size=8,
                                            max_len=max_len))

        # check results. don't expect it to have learned anything meaningful
        # on 5 instances.
        predictions = cnn.predict(estimator, data.input_function(TRAIN_CSV,
                                                                 vocab,
                                                                 classes,
                                                                 batch_size=8,
                                                                 max_len=max_len))
        for p in predictions:
            assert p[0] >= 0.0
            assert p[0] <= 1.0
            assert p[1] >= 0
            assert p[1] <= 1
