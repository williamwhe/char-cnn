# Char-CNN

[![Build Status](https://travis-ci.org/purzelrakete/char-cnn.png?branch=master)](https://travis-ci.org/purzelrakete/char-cnn)

A Keras implementation of [Character-level Convolutional Networks for Text Classification Zhang and LeCun](https://arxiv.org/abs/1509.01626).


## Installation

```bash
pip install char-cnn
```

## Usage
Determine a sample size first:

```python
sample=0.1
```

Now start learning.

```python
from charcnn import cnn, data

# configure
vocab = list(string.printable)
classes = data.dbpedia_classes()
max_len = 1014

# load data
xtrain, ytrain, xtest = data.dbpedia(sample=sample, dataset_source=data.DATA_CLOUD_URL)

# preprocess data
xtrain = data.encode_features(xtrain, vocab, max_len=max_len)
ytrain = data.encode_labels(ytrain, classes)
xtest = data.encode_features(xtest, vocab, max_len=max_len)

# train
model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, len(classes)))
estimator = cnn.estimator(model)
history = cnn.train(estimator, xtrain, ytrain)

print(history.history)
```

You can observe progress using Tensorboard by running

```bash
tensorboard --logdir logs\
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
