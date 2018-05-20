# Char-CNN

[![Build Status](https://travis-ci.org/reflectionlabs/char-cnn.png?branch=master)](https://travis-ci.org/reflectionlabs/char-cnn)

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
