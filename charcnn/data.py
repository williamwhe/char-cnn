# -*- coding: utf-8 -*-

"""
Features, Preprocessing and Datasets, as described in:

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)

"""

import tensorflow as tf

# gzipped trainset file on cloud storage
DATA_CLOUD_TRAINSET = 'gs://char-cnn-datsets/dbpedia/train.csv.gz'

# unknown character integer encoding
UNK = 1

# non printable charcter. hacks around broken utf-8 string_split, see the
# comments in `input_function`.
SPLIT_CHAR = '\a'

# reserve a range from 0. unk: 1, padding: 0.
N_VOCAB_RESERVED = 2


def input_function(file_name,
                   vocab,
                   classes,
                   max_len=1014,
                   shuffle=False,
                   repeat_count=1,
                   batch_size=1,
                   shuffle_buffer_size=1):
    """
    Featurized examples.

    The character splitting hack is due to this open tensorflow bug:

        https://github.com/tensorflow/tensorflow/pull/12971.

    To work around this, we interleave the string with a non printable
    character (BEEP). This character must consequently never be present
    in the source material. This character was chosen because text is highly
    unlikely to include BEEP characters, and also because it is < 128,
    which is required to make this hack work.
    """

    # map into [1,n], leaving 0, n_vocab_reserved free
    vocab_mapped = list(range(N_VOCAB_RESERVED, len(vocab) + N_VOCAB_RESERVED))

    # total vocab size
    n_vocab = len(vocab_mapped) + N_VOCAB_RESERVED

    # number of classes
    n_classes = len(classes)

    def fn():
        d = tf.contrib.lookup.KeyValueTensorInitializer(vocab,
                                                        vocab_mapped,
                                                        key_dtype=tf.string,
                                                        value_dtype=tf.int32)
        table = tf.contrib.lookup.HashTable(d, UNK)

        ds = (tf.data.TextLineDataset(file_name, compression_type='GZIP')
              .map(lambda line: tf.decode_csv(line, [[-1], [''], ['']]))
              .map(lambda y, title, abstract: (title + abstract, y))
              .map(lambda x, y: (tf.regex_replace(x, '.', '\\0%s' % SPLIT_CHAR), y))
              .map(lambda x, y: (tf.string_split([x], delimiter=SPLIT_CHAR).values, y))
              .map(lambda x, y: (x[0:max_len], y))
              .map(lambda x, y: (table.lookup(x), y))
              .map(lambda x, y: (x, tf.one_hot(y, n_classes)))
              .map(lambda x, y: (tf.one_hot(x, n_vocab), y))
              .map(lambda x, y: ({'chars_input': x}, y))
              .repeat(repeat_count))

        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        return ds.padded_batch(batch_size,
                               padded_shapes=({
                                   'chars_input': [max_len, n_vocab]}, [n_classes]))

    return fn


def input_generator(input_fn):
    """
    Evaluated tensors.
    """

    ds = input_fn()
    iterator = ds.make_initializable_iterator()
    init_op = iterator.initializer
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(init_op)

        while True:
            try:
                yield sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break
