# -*- coding: utf-8 -*-

import os
import string
import tempfile

import numpy as np
import six
import tensorflow as tf

from charcnn import data

TRAIN_CSV = 'data/test/train.csv.gz'


class TestInputFunction:

    def test_input_function_features(self):
        input_fn = data.input_function(TRAIN_CSV,
                                       list('ABCDbdeghilmnosy ,'),
                                       range(3),
                                       batch_size=1,
                                       max_len=10)

        batches = list(data.input_generator(input_fn))
        assert len(batches) == 6

        # unpack one more layer because this is a batch of examples, len 1
        got = batches[4][0]['chars_input'][0]
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

    def test_input_function_labels(self):
        input_fn = data.input_function(TRAIN_CSV,
                                       list('ABCDbdeghilmnosy ,'),
                                       range(3),
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

    def test_input_function_resulting_dimension(self):
        vocab, classes = ['a'], range(1)
        input_fn = data.input_function(TRAIN_CSV,
                                       vocab,
                                       classes,
                                       batch_size=10,
                                       max_len=10)

        ds = input_fn()
        iterator = ds.make_initializable_iterator()
        init_op = iterator.initializer
        next_element = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(init_op)
            shape = next_element[0]['chars_input'].shape

            # batch dimension
            assert shape[0].value is None

            # length dimension
            assert shape[1].value is 10

            # this is UNK and PAD. needs to be added to the vocab size.
            n_vocab_reserved = 2

            # vocab dimension
            assert shape[2].value is len(vocab) + n_vocab_reserved

    def test_utf8_strings(self):
        input_fn = data.input_function(TRAIN_CSV,
                                       [six.text_type(u'ä')],
                                       range(3),
                                       batch_size=1,
                                       max_len=10)

        batches = list(data.input_generator(input_fn))
        assert len(batches) == 6

        # "hi"," ginnä"
        got = batches[0][0]['chars_input'][0]

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
        input_fn = data.input_function(TRAIN_CSV,
                                       list('ABCDbdeghilmnosy ,'),
                                       range(3),
                                       batch_size=1,
                                       max_len=1)

        batches = list(data.input_generator(input_fn))
        assert len(batches) == 6

        # "simon"," hi"
        got = batches[1][0]['chars_input'][0]

        # "simon", " hi", but with max_len 1. vocab length including unk and
        # padding is 20. offset of s is 14 + unk + pad = 16 (starting at 0).
        want = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])

        assert np.array_equal(got, want)
