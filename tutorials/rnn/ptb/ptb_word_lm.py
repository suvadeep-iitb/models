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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse as ap

import numpy as np
import tensorflow as tf

import reader

logging = tf.logging

flags = tf.flags
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large, custom")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored")
flags.DEFINE_string("save_path", None,
                    "Model output directory")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_float("init_scale", 0.1, 
                   "Range for the uniform random initializer")
flags.DEFINE_float("learning_rate", 1.0,
                   "Initial learning rate")
flags.DEFINE_float("max_grad_norm", 5,
                   "Max norm for gradient truncation")
flags.DEFINE_integer("num_layers", 2,
                     "Number of layers for recurent neural network")
flags.DEFINE_integer("num_steps", 20,
                     "Context size for the language model")
flags.DEFINE_integer("hidden_size", 200,
                     "Size of the hidden layers")
flags.DEFINE_integer("max_epoch", 4,
                     "Maximum number of epoches with initial learning rate")
flags.DEFINE_integer("max_max_epoch", 20,
                     "Tolal number of epoches")
flags.DEFINE_float("keep_prob", 1.0,
                   "Keep probability for drop out")
flags.DEFINE_float("lr_decay", 0.70,
                   "Multiplier for learning rate decay after max_epoch iterations")
flags.DEFINE_integer("batch_size", 20,
                     "Batch size")
flags.DEFINE_integer("vocab_size", 10000,
                     "Vocabulary size for the language model")
flags.DEFINE_float("exp", 1.0,
                   "Exponent for the dot-product similarity measure")
flags.DEFINE_integer("num_neg_samples", 0,
                     "Number of negative samples. 0 for no down-sampling")
flags.DEFINE_string("loss_func", "logistic",
                    "Loss function to be used. Can be one of (logistic/softmax)")
FLAGS=flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, unigrams, vocab_size, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.num_sampled = num_sampled = config.num_neg_samples
    self.input_data, self.targets, self.neg_samples, self.split_wgts = reader.ptb_producer(
        data, unigrams, batch_size, num_steps, 1, num_sampled, vocab_size, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    num_neg_samples = config.num_neg_samples

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w)
    sign_logits = tf.sign(logits)
    logits = tf.multiply(sign_logits, tf.pow(tf.abs(logits), config.exp))

    perp = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
               [logits],
               [tf.reshape(input_.targets, [-1])],
               [tf.ones([batch_size * num_steps], dtype=data_type())],
               average_across_timesteps=False)

    self._perp_op = tf.reduce_sum(perp) / batch_size

    self._splitted_perp_op = []
    for sw in input_.split_wgts:
        pr = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                  [logits],
                  [tf.reshape(input_.targets, [-1])],
                  [tf.reshape(sw, [-1])],
                  average_across_timesteps=False)
        pr = tf.reduce_sum(pr) / batch_size
        self._splitted_perp_op.append(pr)

    if config.loss_func == 'logistic':
        labels = tf.one_hot(tf.reshape(input_.targets, [-1]), vocab_size)
        self._cost = cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                                          labels=labels,
                                          logits=logits)) / batch_size
    elif config.loss_func == 'softmax':
        self._cost = cost = self._perp_op
    else:
        print('Loss function should be either "logistic" or "softmax"')

    self._final_state = state

    self._grad_op = tf.constant(0)

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    #optimizer = tf.train.AdamOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._grad_op = (tf.reduce_mean(tf.abs(tf.gather(tf.gradients(cost, [softmax_w])[0], tf.reshape(input_.targets, [-1]), axis=1))), tf.reduce_mean(1-tf.gather(logits, tf.reshape(input_.targets, [-1]), axis=1)), tf.reduce_mean(tf.abs(tf.gradients(cost, [softmax_w])[0])))
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def perplexity_op(self):
    return self._perp_op

  @property
  def splitted_perplexity_op(self):
    return self._splitted_perp_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 1
  max_max_epoch = 20
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  exp = 1.0
  num_neg_samples = 0
  loss_func = 'softmax'


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  exp = 1.0
  num_neg_samples = 0
  loss_func = 'softmax'


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  exp = 1.0
  num_neg_samples = 0
  loss_func = 'softmax'


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  exp = 1.0
  num_neg_samples = 20
  loss_func = 'softmax'


class CustomConfig(object):
  """Custom config."""
  init_scale = FLAGS.init_scale
  learning_rate = FLAGS.learning_rate
  max_grad_norm = FLAGS.max_grad_norm
  num_layers = FLAGS.num_layers
  num_steps = FLAGS.num_steps
  hidden_size = FLAGS.hidden_size
  max_epoch = FLAGS.max_epoch
  max_max_epoch = FLAGS.max_max_epoch
  keep_prob = FLAGS.keep_prob
  lr_decay = FLAGS.lr_decay
  batch_size = FLAGS.batch_size
  vocab_size = FLAGS.vocab_size
  exp = FLAGS.exp
  num_neg_samples = FLAGS.num_neg_samples
  loss_func = FLAGS.loss_func


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  perplexity = 0.0
  splitted_perp = [0.0]*len(model._input.split_wgts)
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "perplexity": model.perplexity_op,
      "grad": model._grad_op,
      "splitted_perp": model.splitted_perplexity_op,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    perp = vals["perplexity"]
    state = vals["final_state"]
    sp_perp = vals["splitted_perp"]
     
    costs += cost
    perplexity += perp
    for i, sp in enumerate(sp_perp):
      splitted_perp[i] += sp

    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f loss: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, costs / iters,
             iters * model.input.batch_size / (time.time() - start_time)))

  splitted_perp = [np.exp(sp/iters) for sp in splitted_perp]
  return costs/iters, np.exp(perplexity/iters), splitted_perp


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  elif FLAGS.model == "custom":
    return CustomConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _, unigrams = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, unigrams=unigrams, 
                             vocab_size=config.vocab_size, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, unigrams=unigrams, 
                             vocab_size=config.vocab_size, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, unigrams=unigrams, 
                            vocab_size=config.vocab_size, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        _, train_perplexity, train_sp_perp = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        train_sp_perp = [str(sp) for sp in train_sp_perp]
        print("Epoch: %d Train Perplexity: %.3f (%s)" % (i + 1, train_perplexity, '/'.join(train_sp_perp)))

        _, valid_perplexity, valid_sp_perp = run_epoch(session, mvalid)
        valid_sp_perp = [str(sp) for sp in valid_sp_perp]
        print("Epoch: %d Valid Perplexity: %.3f (%s)" % (i + 1, valid_perplexity, '/'.join(valid_sp_perp)))

        _, test_perplexity, test_sp_perp = run_epoch(session, mtest)
        test_sp_perp = [str(sp) for sp in test_sp_perp]
        print("Epoch: %d Test Perplexity: %.3f (%s)" % (i + 1, test_perplexity, '/'.join(test_sp_perp)))

        if FLAGS.save_path and (i+1) % 5 == 0:
          save_path = FLAGS.save_path + '_' + str(i+1)
          print("Saving model to %s." % save_path)
          sv.saver.save(session, save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
