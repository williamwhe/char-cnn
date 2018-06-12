# -*- coding: utf-8 -*-

"""
Dataset for the cnn. Input a compressed csv with two columns and no header,
where the first column is the class and the second column is a piece of text.

The functions in here will convert the text to a representation the cnn can
use for learning, using the Tensorflow Dataset API.

"""

import tensorflow as tf

# gzipped trainset file on cloud storage
DATA_CLOUD_TRAINSET = 'gs://reflectionlabs/dbpedia/train.csv.gz'

# padding char integer encoding
PADDING = 0

# unknown character integer encoding
UNK = 1

# non printable charcter. hacks around broken utf-8 string_split, see the
# comments in `input_fn`.
SPLIT_CHAR = '\a'

# reserve a range from 0. unk: 1, padding: 0.
N_VOCAB_RESERVED = 2

# length of document. default value used in the paper
DEFAULT_MAX_LEN = 1014


def encode_features(strings_tensor, table, n_vocab, max_len):
    """
    Given a string tensor, generate a one hot representation for the model.

    The character splitting hack is due to this open tensorflow bug:

        https://github.com/tensorflow/tensorflow/pull/12971.

    To work around this, we interleave the string with a non printable
    character (BEEP). This character must consequently never be present
    in the source material. This character was chosen because text is highly
    unlikely to include BEEP characters, and also because it is < 128,
    which is required to make this hack work.
    """

    ret = tf.regex_replace(strings_tensor, '.', '\\0%s' % SPLIT_CHAR)
    ret = tf.string_split(ret, delimiter=SPLIT_CHAR)
    ret = table.lookup(ret)
    ret = tf.sparse_tensor_to_dense(ret, default_value=0)
    ret = ret[:, 0:max_len]
    ret = tf.one_hot(ret, n_vocab)

    return ret


def encode_labels(integers_tensor, n_classes):
    """
    Given integral dense class ids, generate a one hot representation.
    """

    ret = tf.one_hot(integers_tensor, n_classes)

    return ret


def mappings(vocab):
    """
    Mappings for character-wise one hot encoding of the input text.

    Returns:
      table:     a lookup table for characters. maps into [1,n],
                 leaving 0, n_vocab_reserved free
      n_vocab:   number of charaters in the vocab
    """

    mapped = list(range(N_VOCAB_RESERVED, len(vocab) + N_VOCAB_RESERVED))
    n_vocab = len(mapped) + N_VOCAB_RESERVED

    d = tf.contrib.lookup.KeyValueTensorInitializer(vocab,
                                                    mapped,
                                                    key_dtype=tf.string,
                                                    value_dtype=tf.int32)

    return tf.contrib.lookup.HashTable(d, UNK), n_vocab


def pad_features(x, n_vocab, max_len):
    """
    Pad out features so we have a fixed sized representation as well as
    static Tensor Dimensions.
    """

    to_pad = max_len - tf.shape(x)[1]  # longest str in this batch + to_pad
    ret = tf.pad(x, [[0, 0], [0, to_pad], [0, 0]], 'CONSTANT')
    ret.set_shape([None, max_len, n_vocab])

    return ret


def pad_labels(y, n_classes):
    """
    Sets static shape information on the tensor. There's no need to actually
    pad, since all labels will have the same number of dimensions in the
    response vector.
    """

    y.set_shape([None, n_classes])

    return y


def input_fn(file_name,
             vocab,
             classes,
             max_len=DEFAULT_MAX_LEN,
             shuffle=False,
             repeat_count=1,
             batch_size=1,
             shuffle_buffer_size=None):
    """
    Featurized examples.
    """

    # number of classes
    n_classes = len(classes)

    # shuffle buffer
    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size * 2 + 1

    def fn():
        # mapping table and counts
        table, n_vocab = mappings(vocab)

        # transform so we get (text, label) records.
        ds = (tf.data.TextLineDataset(file_name, compression_type='GZIP')
              .map(lambda line: tf.decode_csv(line, [[-1], [''], ['']]))
              .map(lambda y, title, abstract: (title + abstract, y))
              .repeat(repeat_count))

        # shuffle before batching, otherwise we would be shuffling batches
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # batch here so that encode_features and encode_labels always operate
        # on batches. this is so that they can be reused in the serving input
        # receiver function.
        ds = ds.batch(batch_size)

        # encode batches
        ds = (ds
              .map(lambda x, y: (encode_features(x, table, n_vocab, max_len), y))
              .map(lambda x, y: (x, encode_labels(y, n_classes))))

        # prefetch one batch onto the cpu to increase gpu utilization
        ds = ds.prefetch(batch_size)

        # introduce padding and features dict
        return (ds
                .map(lambda x, y: (pad_features(x, n_vocab, max_len), y))
                .map(lambda x, y: (x, pad_labels(y, n_classes)))
                .map(lambda x, y: ({'chars_input': x}, y)))

    return fn


def serving_input_receiver_fn(vocab,
                              batch_size,
                              max_len=DEFAULT_MAX_LEN):
    """
    Serving function. Receives a single batch of batch_size.
    """

    def fn():
        # mapping table and counts
        table, n_vocab = mappings(vocab)

        # input examples. lines of plain text, batch size unknown, since there
        # may be less than batch_size examples sometimes.
        examples = tf.placeholder(dtype=tf.string, shape=[None])

        # incoming prediction queries
        feature_placeholders = {
            'examples': examples
        }

        # apply the same transforms we used in the input_fn
        chars_input = encode_features(examples, table, n_vocab, max_len)

        # pad out features
        chars_input = pad_features(chars_input, n_vocab, max_len)

        # transformed for model usage
        features = {
            'chars_input': chars_input
        }

        return tf.estimator.export.ServingInputReceiver(features,
                                                        feature_placeholders)

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
