# -*- coding: utf-8 -*-

import os
import string
import tempfile

import numpy as np
import six
import tensorflow as tf

from charcnn import data

TRAIN_CSV = 'data/test/train.csv.gz'


class TestServingData:
    """
    Test tensorflow serving functions
    """

    def test_serving_input_receiver_fn_returns_spec(self):
        serving_input_receiver = data.serving_input_receiver_fn(['a'], 64)()
        assert serving_input_receiver


class TestInputFunction:

    def test_input_fn_resulting_dimension(self):
        vocab, classes, max_len = ['a'], list(range(1)), 10
        input_fn = data.input_fn(TRAIN_CSV,
                                 vocab,
                                 classes,
                                 batch_size=1,
                                 max_len=max_len)

        ds = input_fn()
        iterator = ds.make_initializable_iterator()
        init_op = iterator.initializer
        next_element = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(init_op)
            sess.run(next_element)

            shape = next_element[0]['chars'].shape

            # batch dimension
            assert shape[0].value is None

            # length dimension
            assert shape[1].value is max_len

            # this is UNK and PAD. needs to be added to the vocab size.
            n_vocab_reserved = 2

            # vocab dimension
            assert shape[2].value is len(vocab) + n_vocab_reserved

    def test_input_fn_labels(self):
        input_fn = data.input_fn(TRAIN_CSV,
                                 list('ABCDbdeghilmnosy ,'),
                                 list(range(3)),
                                 batch_size=6,
                                 max_len=10)

        batches = list(data.input_generator(input_fn))
        assert len(batches) == 1

        got = batches[0][1].astype(np.float32)
        want = np.array([[1., 0., 0.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [1., 0., 0.],
                         [0., 0., 1.]])

        assert np.array_equal(got, want)

    def test_input_fn_features(self):
        input_fn = data.input_fn(TRAIN_CSV,
                                 list('ABCDbdeghilmnosy ,'),
                                 list(range(3)),
                                 batch_size=1,
                                 max_len=10)

        batches = list(data.input_generator(input_fn))
        assert len(batches) == 6

        # unpack one more layer because this is a batch of examples, len 1
        got = batches[4][0]['chars'][0]
        want = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        assert np.array_equal(got, want)

    def test_utf8_strings(self):
        input_fn = data.input_fn(TRAIN_CSV,
                                 [six.text_type(u'ä')],
                                 list(range(3)),
                                 batch_size=1,
                                 max_len=10)

        batches = list(data.input_generator(input_fn))
        assert len(batches) == 6

        # "hi"," ginnä"
        got = batches[0][0]['chars'][0]

        # "hi"," ginnä". but with only 'ä' in the vocabulary, we have
        # [pad, unk, 'ä'] vocab. so everything unknown until the last letter,
        # and then padded to 10 chars so 2 more pad rows.
        want = np.array([[0., 1., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [0., 0., 0.],
                         [0., 0., 0.]])

        assert np.array_equal(got, want)

    def test_strings_longer_than_max_len(self):
        input_fn = data.input_fn(TRAIN_CSV,
                                 list('ABCDbdeghilmnosy ,'),
                                 list(range(3)),
                                 batch_size=1,
                                 max_len=1)

        batches = list(data.input_generator(input_fn))
        assert len(batches) == 6

        # "simon"," hi"
        got = batches[1][0]['chars'][0]

        # "simon", " hi", but with max_len 1. vocab length including unk and
        # padding is 20. offset of s is 14 + unk + pad = 16 (starting at 0).
        want = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])

        assert np.array_equal(got, want)
