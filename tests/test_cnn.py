# -*- coding: utf-8 -*-

import os
import string

import numpy as np
import tensorflow as tf

from charcnn import cnn
from charcnn import data

TRAIN_CSV = 'data/test/train.csv.gz'


class TestIntegration:
    """
    Integration test for Estimator
    """

    # tests
    #

    def setup_method(self):
        tf.reset_default_graph()

    def test_train_and_predict(self):
        vocab = list('ABCDbdeghilmnosy ,')
        classes = list(range(3))
        params = {'classes': classes, 'vocab': vocab}
        max_len = 300

        estimator = tf.estimator.Estimator(model_fn=cnn.model_fn,
                                           params=params,
                                           config=tf.estimator.RunConfig())

        train_input_fn = data.input_fn(TRAIN_CSV,
                                       vocab,
                                       classes,
                                       batch_size=8,
                                       shuffle=True,
                                       max_len=max_len,
                                       repeat_count=1)

        test_input_fn = data.input_fn(TRAIN_CSV,
                                      vocab,
                                      classes,
                                      batch_size=8,
                                      shuffle=False,
                                      max_len=max_len,
                                      repeat_count=1)

        # train
        estimator.train(train_input_fn)

        # predict
        preds = list(estimator.predict(test_input_fn))

        assert len(preds) == 6
        assert np.all(1.0 - np.array([sum(p['probabilities']) for p in preds]) < 1e-7)
        assert set([p['prediction_index'] for p in preds]).issubset(set(classes))

        for predicted_class in [p['prediction'] for p in preds]:
            assert predicted_class in classes
