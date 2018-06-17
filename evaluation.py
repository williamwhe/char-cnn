#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Notebook

"""

import subprocess

import numpy as np
import pandas as pd

# base url for prediction results
PREDICTIONS_BASE_URL = 'gs://reflectionlabs/predict'


def gs_ls(url):
    """
    gsutil ls
    """

    cmd = ['gsutil', 'ls', url]

    return (subprocess
            .check_output(cmd)
            .strip()
            .split('\n'))


def gs_cat(url):
    """
    gsutil cat
    """

    cmd = ['gsutil', 'cat', url]

    return subprocess.check_output(cmd)


def df_predictions(job_name, verbose=False):
    """
    predictions for a given run
    """

    predictions = []
    for f in gs_ls('%s/%s' % (PREDICTIONS_BASE_URL, job_name)):
        if 'results' in f:  # filter out error files
            if verbose:
                print('getting %s' % f)

            res = pd.read_json(gs_cat(f), lines=True)
            predictions.append(res)

    return pd.concat(predictions)
