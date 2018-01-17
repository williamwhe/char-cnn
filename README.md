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
sample=0.8
```

Now start learning.

```python
from keras.callbacks import TensorBoard
from charcnn import cnn, data

xtrain, ytrain, xtest = data.dbpedia(sample=sample, dataset_source=data.DATA_CLOUD_URL)
xtrain, ytrain, xtest, vocab, max_len, n_classes = data.preprocess(xtrain, ytrain, xtest, max_len=1014)

callbacks = [TensorBoard(write_images=True)]
model = cnn.compiled(cnn.char_cnn(len(vocab), max_len, n_classes))
history = cnn.fit(model, xtrain, ytrain, callbacks)

print(history.history)
```

You can observe progress using Tensorboard by running

```bash
tensorboard --logs logs\
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
