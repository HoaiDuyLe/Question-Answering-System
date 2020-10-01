from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
from inference import model_fn_builder
import tensorflow as tf

def serving_input_fn():
  inputs = {
    "unique_ids": tf.placeholder(tf.int64, shape = [None], name='unique_ids'),
    "input_ids": tf.placeholder(tf.int64, shape = [None, FLAGS.max_seq_length], name='input_ids'),
    "input_mask": tf.placeholder(tf.int64, shape =[None, FLAGS.max_seq_length], name='input_mask'),
    "segment_ids": tf.placeholder(tf.int64, shape =[None, FLAGS.max_seq_length], name='segment_ids')}        
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

if __name__ == '__main__':
    estimator._export_to_tpu = False
    estimator.export_savedmodel('export_t', serving_input_fn)
