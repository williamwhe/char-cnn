#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Control training on google cloud ml from the command line.

"""

import os
import subprocess
import tempfile
import time

import fire

# google cloud region
DEFAULT_REGION = 'us-east1'

# google cloud bucket
DEFAULT_BUCKET = 'reflectionlabs'

# tensorflow runtime version
TF_RUNTIME_VERSION = '1.8'

# maximum document length
MAX_LEN = 1014


def production(project,
               bucket,
               train_files,
               test_files,
               vocab_file,
               classes_file,
               verbose=True):
    """
    Do a production training run
    """

    job_name = 't%s' % int(round(time.time()))
    job_dir = 'gs://%s/train/%s' % (bucket, job_name)

    cmd = ['gcloud', 'ml-engine', 'jobs', 'submit', 'training', job_name,
           '--module-name', 'charcnn.task',
           '--package-path', 'charcnn',
           '--job-dir', job_dir,
           '--config', 'train.yml',
           '--project', project,
           '--region', DEFAULT_REGION,
           '--runtime-version', TF_RUNTIME_VERSION,
           '--',
           '--train-files', train_files,
           '--test-files', test_files,
           '--max-len', str(MAX_LEN),
           '--vocab-file', vocab_file,
           '--classes-file', classes_file,
           '--job-dir', job_dir
           ]

    if verbose:
        print(' '.join(cmd))

    subprocess.call(cmd)


def development(train_files,
                test_files,
                vocab_file,
                classes_file,
                verbose=True):
    """
    Do a local training run
    """

    job_dir = str(tempfile.mkdtemp())

    cmd = ['gcloud', 'ml-engine', 'local', 'train',
           '--module-name', 'charcnn.task',
           '--package-path', 'charcnn',
           '--job-dir', job_dir,
           '--',
           '--train-files', train_files,
           '--test-files', test_files,
           '--train-batch-size', '4',
           '--test-batch-size', '4',
           '--max-len', str(MAX_LEN),
           '--vocab-file', vocab_file,
           '--classes-file', classes_file,
           ]

    if verbose:
        print(' '.join(cmd))

    subprocess.call(cmd)


def tensorboard(job_name,
                bucket=DEFAULT_BUCKET,
                verbose=True):
    """
    Launch a Tensorboard to visualize the given run.
    """

    job_dir = 'gs://%s/train/%s' % (bucket, job_name)

    cmd = ['tensorboard', '--logdir', job_dir]

    if verbose:
        print(' '.join(cmd))

    subprocess.call(cmd)


if __name__ == '__main__':
    fire.Fire({
        'development': development,
        'production': production,
        'tensorboard': tensorboard
    })
