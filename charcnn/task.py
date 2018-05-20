#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Train and evaluate the model

import argparse
import os

import pandas as pd
import tensorflow as tf

from charcnn import cnn
from charcnn import data


def train_and_evaluate(train_files,
                       test_files,
                       vocab,
                       classes,
                       max_len,
                       epochs,
                       train_batch_size,
                       test_batch_size,
                       job_dir):
    """
    Run the training and evaluation using estimators, then save the model.
    """

    # set up the training data
    train_input = data.input_function(train_files,
                                      vocab,
                                      classes,
                                      max_len=max_len,
                                      shuffle=True,
                                      batch_size=train_batch_size,
                                      repeat_count=epochs)

    # set up the test data
    test_input = data.input_function(train_files,
                                     vocab,
                                     classes,
                                     max_len=max_len,
                                     shuffle=False,
                                     batch_size=test_batch_size,
                                     repeat_count=epochs)

    # construct the estimator
    estimator = cnn.build(vocab,
                          max_len,
                          classes,
                          job_dir=job_dir)

    # train and test the estimator
    tf.estimator.train_and_evaluate(estimator,
                                    tf.estimator.TrainSpec(train_input),
                                    tf.estimator.EvalSpec(test_input))


def parse_args():
    """
    Parse command line args
    """

    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--test-files',
        help='GCS or local paths to evaluation data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--vocab-file',
        help='UTF-8 characters in the dictionary are in this file, one per line',
        required=True
    )
    parser.add_argument(
        '--classes-file',
        help="A list of classes. Headerless CSV, id,name.",
        required=True
    )
    parser.add_argument(
        '--max-len',
        help='Maximum document length',
        type=int,
        default=1014
    )
    parser.add_argument(
        '--epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --n-epochs are specified,
        the training job will run for --max-steps or --n-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
        type=int,
        default=5
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=32
    )
    parser.add_argument(
        '--test-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=32
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
        help='Set logging verbosity'
    )

    return vars(parser.parse_args())


def init_logging(verbosity):
    tf.logging.set_verbosity(verbosity)
    min_log_level = str(tf.logging.__dict__[verbosity] / 10)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = min_log_level


if __name__ == '__main__':
    config = parse_args()
    init_logging(config['verbosity'])

    # read out configuration
    df_vocab = pd.read_csv(config['vocab_file'], header=None, names=['char'])
    df_classes = pd.read_csv(config['classes_file'], header=None, names=['idx', 'name'])

    # grab the respective columns from the csv
    vocab = list(df_vocab.char)
    classes = list(df_classes.idx)
    class_names = list(df_classes.name)

    # start training and evaluation
    train_and_evaluate(config['train_files'],
                       config['test_files'],
                       vocab,
                       classes,
                       config['max_len'],
                       config['epochs'],
                       config['train_batch_size'],
                       config['test_batch_size'],
                       config['job_dir'])
