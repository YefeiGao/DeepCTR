#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob

import tensorflow as tf
import numpy as np
import re
from multiprocessing import Pool as ThreadPool

flags = tf.app.flags
FLAGS = flags.FLAGS
LOG = tf.logging

tf.app.flags.DEFINE_string("input_dir", "./", "input dir")
tf.app.flags.DEFINE_string("output_dir", "./", "output dir")
tf.app.flags.DEFINE_integer("threads", 2, "threads num")

max_varlen = 5
sparse_feat_fields = {}
varlen_feat_fields = {}
with open("../bin/field_feature.txt", 'r') as ff:
  for line in ff.readlines():
    field_id, _, field_name, _, bucket_size, feature_class = line.strip().split("\t")
    if feature_class == "multi-cat":
      varlen_feat_fields[field_name] = int(field_id)
    else:
      sparse_feat_fields[field_name] = int(field_id)

def gen_tfrecords(in_file):
  basename = os.path.basename(in_file) + ".tfrecord"
  out_file = os.path.join(FLAGS.output_dir, basename)
  tfrecord_out = tf.io.TFRecordWriter(out_file)
  with open(in_file) as fi:
    for line in fi:
      fields = line.strip().split('\t')
      if len(fields) != 53:
        continue
      # 1 label
      label = [int(fields[0])]

      feature = {
        "prediction_layer": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
      }

      # for sparse feats
      for feat_name, field_id in sparse_feat_fields.items():
        feat_val = np.array([fields[field_id]], np.int)
        feature.update({feat_name: tf.train.Feature(int64_list=tf.train.Int64List(value=feat_val))})

      # for varlen sparse feats
      for feat_name, field_id in varlen_feat_fields.items():
        multi_feats = fields[field_id].split(":")
        if len(multi_feats) >= max_varlen:
          multi_feats = multi_feats[:max_varlen]
        else:
          multi_feats.extend([0] * (max_varlen - len(multi_feats)))
        feat_val = np.array(multi_feats, np.int)
        feature.update({feat_name: tf.train.Feature(int64_list=tf.train.Int64List(value=feat_val))})

      # serialized to Example
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      serialized = example.SerializeToString()
      tfrecord_out.write(serialized)
      # num_lines += 1
      # if num_lines % 10000 == 0:
      #    print("Process %d" % num_lines)
  tfrecord_out.close()


def main(_):
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  file_list = glob.glob(os.path.join(FLAGS.input_dir, "*"))
  print("total files: %d" % len(file_list))

  pool = ThreadPool(FLAGS.threads)  # Sets the pool size
  pool.map(gen_tfrecords, file_list)
  pool.close()
  pool.join()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
