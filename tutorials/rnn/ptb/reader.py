# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, counts = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id, counts


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id, unigrams = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary, unigrams


def get_label_bucket_masks(vocab_size):
  label_buckets = [range(10), range(10, 100), range(100, 1000), range(1000, vocab_size)]
  label_bucket_masks = []
  for lb in label_buckets:
    lbm = np.zeros((vocab_size), dtype=np.float32)
    lbm[lb] = 1.0
    label_bucket_masks.append(tf.convert_to_tensor(lbm))
  return label_bucket_masks
    

def get_split_weights_for_perp(y, unigrams, vocab_size, batch_size):
  lbm_list = get_label_bucket_masks(vocab_size)
  one_hot_y = tf.one_hot(y, vocab_size)
  split_weights = []
  for lbm in lbm_list:
    wgt = tf.matmul(one_hot_y, tf.reshape(lbm, [-1, 1]))
    split_weights.append(tf.reshape(wgt, [-1]))
  return split_weights


def get_neg_samples(batch_size, num_true, num_sampled, vocab_size, true_classes, unigrams):
  unigrams = list(unigrams)
  if len(unigrams) < vocab_size:
    unigrams += [0]*(vocab_size-len(unigrams))
  neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                  true_classes=true_classes,
                  num_true=num_true,
                  num_sampled=num_sampled,
                  unique=False,
                  range_max=vocab_size,
                  distortion=0.75,
                  unigrams=unigrams
               )
  return neg_samples


def ptb_producer(raw_data, unigrams, batch_size, num_steps, num_true, num_sampled, vocab_size, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int64)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    ns = None

    psw_list = get_split_weights_for_perp(tf.reshape(y, [-1]), unigrams, vocab_size, batch_size*num_steps)
    
    for i in range(len(psw_list)):
      psw_list[i] = tf.reshape(psw_list[i], [batch_size, num_steps])

    '''
    if num_sampled > 0:
      y_list = tf.unpack(y, axis=1)
      ns_list = []
      for i in range(num_steps):
        ns = get_neg_samples(batch_size, num_true, num_sampled, vocab_size, y_list[i], unigrams)
        ns_list.append(ns)
    else:
      ns = None
    '''

    return x, y, ns, psw_list

