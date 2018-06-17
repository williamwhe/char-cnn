# Char-CNN

[![Build Status](https://travis-ci.org/reflectionlabs/char-cnn.png?branch=master)](https://travis-ci.org/reflectionlabs/char-cnn)
[![Documentation Status](https://readthedocs.org/projects/char-cnn/badge/?version=latest)](http://char-cnn.readthedocs.io/en/latest/?badge=latest)

A Tensorflow implementation of [Character-level Convolutional Networks for Text Classification Zhang and LeCun](https://arxiv.org/abs/1509.01626).

## Installation

```bash
pip install char-cnn
```

## Usage

```python
from charcnn import cnn
from charcnn import data
import pandas as pd
import tensorflow as tf

df_vocab = pd.read_csv(data.VOCAB_FILE, header=None, names=['char'])
df_classes = pd.read_csv(data.CLASSES_FILE, header=None, names=['idx', 'name'])

vocab = list(df_vocab.char)
classes = list(df_classes.idx)
class_names = list(df_classes.name)
params = {'classes': classes, 'vocab': vocab}

max_len, batch_size, epochs = 1014, 128, 5
save_checkpoints_steps=100
job_dir = '/tmp/char-cnn-job'

train_input = data.input_fn(data.CLOUD_TRAINSET,
                            vocab,
                            classes,
                            max_len=max_len,
                            shuffle=True,
                            batch_size=batch_size,
                            repeat_count=epochs)

test_input = data.input_fn(data.CLOUD_TESTSET,
                           vocab,
                           classes,
                           max_len=max_len,
                           shuffle=False,
                           batch_size=batch_size,
                           repeat_count=epochs)

# configure estimator run
run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=save_checkpoints_steps,
    model_dir=job_dir
)

# construct the estimator
estimator = tf.estimator.Estimator(model_fn=cnn.model_fn,
                                   params=params,
                                   config=run_config)

# train and test the estimator
tf.estimator.train_and_evaluate(estimator,
                                tf.estimator.TrainSpec(train_input, max_steps=max_steps),
                                tf.estimator.EvalSpec(test_input))

# save the model
estimator.export_savedmodel(saved_model_dir,
                            data.serving_input_receiver_fn(vocab, test_batch_size),
                            strip_default_attrs=True)
```

You can observe progress using Tensorboard by running

```bash
tensorboard --logdir logs
```

### Running on Google Cloud ML Engine

You will need to check out the project, install pipenv and `pipenv install
--dev`. After installing the google cloud toolchain and authenticating your
account you can run the following.

To test locally:

```bash
bin/mlengine development \
  --train-files data/test/train.csv.gz \
  --test-files data/test/test.csv.gz \
  --vocab-file https://storage.googleapis.com/reflectionlabs/dbpedia/chars.csv \
  --classes-file https://storage.googleapis.com/reflectionlabs/dbpedia/classes.csv
```

For a full GPU cloud run on dbpedia:

```bash
bin/mlengine production \
  --project char-cnn \
  --bucket reflectionlabs \
  --train-files gs://reflectionlabs/dbpedia/train.csv.gz \
  --test-files gs://reflectionlabs/dbpedia/test.csv.gz \
  --vocab-file https://storage.googleapis.com/reflectionlabs/dbpedia/chars.csv \
  --classes-file https://storage.googleapis.com/reflectionlabs/dbpedia/classes.csv
```

Prediction:

```bash
bin/mlengine prediction \
  --project char-cnn \
  --bucket reflectionlabs \
  --model-name t1529179141 \
  --batch-size 64 \
  --input-paths gs://reflectionlabs/dbpedia/test.csv.gz
```

## Citation

```citation
@article{DBLP:journals/corr/ZhangZL15,
  author    = {Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
  title     = {Character-level Convolutional Networks for Text Classification},
  journal   = {CoRR},
  volume    = {abs/1509.01626},
  year      = {2015},
  url       = {http://arxiv.org/abs/1509.01626},
  archivePrefix = {arXiv},
  eprint    = {1509.01626},
  timestamp = {Wed, 07 Jun 2017 14:41:26 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/ZhangZL15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
