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
import string

from charcnn import cnn
from charcnn import data

vocab = list(string.printable)
classes = range(14)
max_len, batch_size = 1014, 128

estimator = cnn.build(vocab, max_len, classes)
estimator.train(data.input_function(data.DATA_CLOUD_TRAINSET,
                                    vocab,
                                    classes,
                                    batch_size=batch_size,
                                    max_len=max_len))
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
bin/train development \
  --train-files data/test/train.csv.gz \
  --test-files data/test/train.csv.gz \
  --vocab-file https://storage.googleapis.com/reflectionlabs/dbpedia/chars.csv \
  --classes-file https://storage.googleapis.com/reflectionlabs/dbpedia/classes.csv
```

For a full GPU cloud run on dbpedia:

```bash
bin/train production \
  --project char-cnn \
  --bucket reflectionlabs \
  --train-files gs://reflectionlabs/dbpedia/train.csv.gz \
  --test-files gs://reflectionlabs/dbpedia/test.csv.gz \
  --vocab-file https://storage.googleapis.com/reflectionlabs/dbpedia/chars.csv \
  --classes-file https://storage.googleapis.com/reflectionlabs/dbpedia/classes.csv
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
