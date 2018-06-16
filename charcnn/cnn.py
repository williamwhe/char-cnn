#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An implementation of

    Character-level Convolutional Networks for Text Classification
    Zhang and LeCun, 2015 (See https://arxiv.org/abs/1509.01626)

"""

import numpy as np
import json

import tensorflow as tf

from charcnn import data


def char_cnn(features, n_vocab, n_classes, with_dropout=True):
    """
    Char-CNN, see Zhang and LeCun, 2015.
    """

    # reserve for special control characters – eg unknown, padding
    n_vocab = n_vocab + data.N_VOCAB_RESERVED

    if with_dropout:
        dropout_probability = 0.5
    else:
        dropout_probability = 0.0

    def conv(inputs, filters, kernel_size):
        activation_layer = tf.layers.Conv1D(filters=filters,
                                            kernel_size=kernel_size,
                                            padding='same',
                                            activation=tf.nn.relu,
                                            dtype=inputs.dtype.base_dtype)

        activation = activation_layer(inputs)

        tf.summary.histogram('activations', activation)
        tf.summary.histogram('kernel', activation_layer.kernel)
        tf.summary.histogram('bias', activation_layer.bias)

        tf.summary.scalar('activation_non_zeros', tf.count_nonzero(activation_layer.bias))
        tf.summary.scalar('kernel_non_zeros', tf.count_nonzero(activation_layer.bias))
        tf.summary.scalar('bias_non_zeros', tf.count_nonzero(activation_layer.bias))

        return activation

    def pool(activation, pool_size):
        mp = tf.layers.max_pooling1d(inputs=activation,
                                     pool_size=pool_size,
                                     strides=pool_size)

        return mp

    def dense(features, units, with_dropout=True):
        d = tf.layers.dense(inputs=features, units=units)
        if with_dropout:
            d = tf.layers.dropout(d, dropout_probability)

        return d

    # char-cnn
    #
    with tf.name_scope('block-1'):
        c1 = conv(features['chars'], filters=256, kernel_size=7)
        c1 = pool(c1, pool_size=3)

    with tf.name_scope('block-2'):
        c2 = conv(c1, filters=256, kernel_size=7)
        c2 = pool(c2, pool_size=3)

    with tf.name_scope('block-3'):
        c3 = conv(c2, filters=256, kernel_size=3)
        c4 = conv(c3, filters=256, kernel_size=3)
        c5 = conv(c4, filters=256, kernel_size=3)
        c6 = conv(c5, filters=256, kernel_size=3)
        c6 = pool(c6, pool_size=3)

    with tf.name_scope('dense'):
        f1 = tf.layers.flatten(inputs=c6)
        d1 = dense(f1, units=1024)
        d2 = dense(d1, units=1024)
        logits = dense(d2, units=n_classes, with_dropout=False)

    return logits


def estimator(run_config, hparams):
    """
    Returns an Estimator.
    """

    return tf.estimator.Estimator(model_fn=model_fn,
                                  params=hparams,
                                  config=run_config)


def model_fn(features, labels, mode, params):
    """
    Estimator model function for prediction, training and evaluation.
    """

    def probability_ops(logits):
        probabilities = tf.nn.softmax(logits, name='classes')
        return probabilities, tf.argmax(probabilities, axis=1)

    def loss_op(labels, logits):
        return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # prediction
    logits = char_cnn(features, len(params['vocab']), len(params['classes']))

    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities, predicted_indices = probability_ops(logits)

        predictions = {
            'prediction_index': predicted_indices,
            'prediction': tf.gather(params['classes'], predicted_indices),
            'probabilities': probabilities
        }

        # add ground truth to the output if it's there
        if 'ground_truth' in features:
            predictions['ground_truth'] = features['ground_truth']

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs={
                'predictions': tf.estimator.export.PredictOutput(predictions)
            })

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        loss = loss_op(labels, logits)
        optimizer = tf.train.AdamOptimizer()

        # get hold of the gradients in order to summarize them
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        # metrics
        tf.summary.scalar('cross_entropy', loss)

        for pair in gradients:
            gradient, variable = pair
            summary_name = ('%s_gradient' % variable.name).replace(':', '_')
            tf.summary.histogram(summary_name, gradient)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)

    # evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities, predicted_indices = probability_ops(logits)
        label_indices = tf.argmax(input=labels, axis=1)
        loss = loss_op(labels, logits)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
                'auroc': tf.metrics.auc(labels, probabilities)
            })
