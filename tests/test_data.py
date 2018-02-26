import os

import numpy as np
import pandas as pd

from charcnn import data


class TestData:
    "Data and feaures"

    def test_encode_features(self):
        xtrain = data.encode_features(lines('data/test/xtrain.txt'),
                                      list('ABCDbdeghilmnosy ,'),
                                      max_len=10)

        # 'hello', padded to max_len = 10 with n_vocab = 18.
        got = xtrain[4].astype(np.float32)
        want = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])

        assert np.array_equal(got, want)

    def test_encode_labels(self):
        ytrain = data.encode_labels(lines('data/test/ytrain.txt'),
                                    ['hi', 'bye', 'unk'])

        got = ytrain.astype(np.float32)
        want = np.array([[1., 0., 0.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.]])

        assert np.array_equal(got, want)


# Testing utilty functions

def lines(filename):
    with open(filename) as f:
        return f.read().splitlines()
