# -*- coding: utf-8 -*-

"""
Features, Preprocessing and Datasets, as described in:

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)


"""

from functools import reduce

import keras as ks
import numpy as np
import pandas as pd


# available when the project is checked out, not when pip installed.
DATA_LOCAL_PATH = 'data'

# remote path from google cloud storage
DATA_CLOUD_URL = 'https://storage.googleapis.com/char-cnn-datsets'


def onehot(features, max_len, vocab_size):
    """
    One-hot encode each letter
    """

    hot = np.zeros((len(features), max_len, vocab_size), dtype=np.bool)
    i = 0
    for line in features:
        j = 0
        for char in line:
            if char != 0:
                hot[i, j, char] = 1.

            j += 1
        i += 1

    return hot


def lookup_table(els):
    "reverse positional index on a list"

    return dict(((c, i) for c, i in zip(els, range(len(els)))))


def encode_features(features, vocab, idx_letters=None, max_len=1014):
    """
    Featurize the text to be classified
    """

    # lookup table
    if idx_letters is None:
        idx_letters = lookup_table(vocab)

    # encode features
    features = [[idx_letters[char] for char in list(line)] for line in features]

    # pad features
    features = ks.preprocessing.sequence.pad_sequences(features, max_len)

    # one hot encode
    return onehot(features, max_len, len(vocab))


def encode_labels(labels, classes, idx_classes=None):
    """
    One hot encode the classes
    """

    # lookup table
    if idx_classes is None:
        idx_classes = lookup_table(classes)

    # encode labels
    labels = [idx_classes[line] for line in labels]

    # one hot encode
    return ks.utils.to_categorical(labels, num_classes=len(classes))


def examples(features, labels, vocab, classes, max_len):
    """
    Generator, given features and labels, emits encoded features and labels.
    """

    # compute lookup tables once
    idx_letters = lookup_table(vocab)
    idx_classes = lookup_table(classes)

    # generate one example at a time
    examples = zip(features, labels)
    for i, (features, label) in enumerate(examples):
        features = encode_features([features], vocab, idx_letters, max_len)
        label = encode_labels([label], classes, idx_classes)
        yield features, label


def dbpedia(sample=None, dataset_source=DATA_LOCAL_PATH):
    """
    DBpedia is a crowd-sourced community effort to extract structured
    information from Wikipedia. The DBpedia ontology dataset is constructed by
    picking 14 nonoverlapping classes from DBpedia 2014. From each of these 14
    ontology classes, we randomly choose 40,000 training samples and 5,000
    testing samples. The fields we used for this dataset contain title and
    abstract of each Wikipedia article.
    """

    names = ['label', 'title', 'body']
    df_train = pd.read_csv(
        dataset_source + '/dbpedia/train.csv.gz',
        header=None,
        names=names)

    df_test = pd.read_csv(
        dataset_source + '/dbpedia/test.csv.gz',
        header=None,
        names=names)

    if sample:
        df_train = df_train.sample(frac=sample)
        df_test = df_test.sample(frac=sample)

    xtrain = df_train['body'].values
    ytrain = df_train['label'].values.astype('int32')
    xtest = df_test['body'].values

    return xtrain, ytrain, xtest


def dbpedia_classes():
    "FIXME(rk): 14 classes are numbered through from zero"

    return map(str, range(14))
