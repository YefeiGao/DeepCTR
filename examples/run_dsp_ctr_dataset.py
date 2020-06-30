import glob
import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

sys.path.append("../")
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.backend import set_session, get_session, count_params
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from deepctr.models import DeepFM, AFM, NFM, DIN
from deepctr.inputs import SparseFeat, VarLenSparseFeat


def parse_args():
  parser = argparse.ArgumentParser(description='training args')
  # ----net----#
  parser.add_argument('--net', dest='net', help='net module', default='pnn', type=str)

  # ----data----#
  parser.add_argument('--data-type', dest='data_type', help='data type like: textline, tfrecords, etc.',
                      default='textline', type=str)
  parser.add_argument('--train-data', dest='train_data', help='training data', default='../data/dsp_ctr/train', type=str)
  parser.add_argument('--test-data', dest='test_data', help='test data', default='../data/dsp_ctr/test', type=str)
  parser.add_argument('--batch-size', dest='batch_size', help='training mini-batch size', default=128, type=int)
  parser.add_argument('--num-epoch', dest='num_epoch', help='the number of training epochs', default=1, type=int)
  parser.add_argument('--train-size', dest='train_size', help='the number of training data', default=10000, type=int)
  parser.add_argument('--test-size', dest='test_size', help='the number of test data', default=1000, type=int)

  # ----model----#
  parser.add_argument('--model-dir', dest='model_dir', help='path to save model checkpoints', default='../model', type=str)
  parser.add_argument('--log-dir', dest='log_dir', help='path to save log files', default='../log', type=str)

  # ----others----#
  parser.add_argument('--feat-config', dest='feat_config', help='feature configure file',
                      default='../bin/field_feature.txt', type=str)
  parser.add_argument('--padding-size', dest='padding_size', help='num for padding varlen features', default=5, type=int)

  args = parser.parse_args()
  return args


def get_feature_info(feature_config_file):
  linear_sparse_features = {}
  sparse_features = {}
  varlen_features = {}
  linear_varlen_features = {}
  with open(feature_config_file, 'r') as ff:
    for line in ff.readlines():
      field_id, _, field_name, _, bucket_size, feature_class, padding_size, wd = line.strip().split("\t")
      if wd == "w&d":
        if bucket_size == "-1":
          bucket_size = 1002
        if feature_class == "multi-cat":
          varlen_features[field_name] = [int(field_id), int(bucket_size), int(padding_size)]
          linear_varlen_features[field_name] = [int(field_id), int(bucket_size), int(padding_size)]
        else:
          sparse_features[field_name] = [int(field_id), int(bucket_size)]
          linear_sparse_features[field_name] = [int(field_id), int(bucket_size)]
      else:
        if bucket_size == "-1":
          bucket_size = 1002
        if feature_class == "multi-cat":
          linear_varlen_features[field_name] = [int(field_id), int(bucket_size), int(padding_size)]
        else:
          linear_sparse_features[field_name] = [int(field_id), int(bucket_size)]
  return sparse_features, varlen_features, linear_sparse_features, linear_varlen_features


def example_parser(example, sparse_feat_ids, varlen_feat_ids, is_training=True):
  columns = tf.string_split([example], '\t')
  label = {"prediction_layer": tf.string_to_number([columns.values[0]], out_type=tf.int64)}
  feature = {}
  sparse_feats = {feat: tf.string_to_number([columns.values[sparse_feat_ids[feat][0]]], out_type=tf.int64)
                  for feat in sparse_feat_ids.keys()}
  feature.update(sparse_feats)
  varlen_feats = {}
  for feat, feat_info in varlen_feat_ids.items():
    multi_cat_feat_padding = tf.zeros([feat_info[2]], tf.int64)
    multi_cat_feat = tf.string_split([columns.values[feat_info[0]]], ":")
    multi_cat_feat = tf.string_to_number(multi_cat_feat.values, out_type=tf.int64)
    multi_cat_feat = tf.concat([multi_cat_feat, multi_cat_feat_padding], 0)[:feat_info[2]]
    varlen_feats[feat] = multi_cat_feat

  feature.update(varlen_feats)
  return feature, label


def get_dataset(files, parse_function, sparse_feats, varlen_feats, data_type, num_parallel_calls=10, batch_size=256):
  print('Parsing', files)
  if data_type == "textline":
    dataset = tf.data.TextLineDataset(files)
  elif data_type == "tfrecords":
    dataset = tf.data.TFRecordDataset(files)
  dataset = dataset.map(lambda line: parse_function(line, sparse_feats, varlen_feats, is_training=True),
                        num_parallel_calls=num_parallel_calls)
  dataset = dataset.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
  dataset.repeat(10)
  try:
    dataset = dataset.batch(batch_size, drop_remainder=True)
  except:
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  dataset = dataset.prefetch(batch_size)
  return dataset


# auc tools 2
def as_keras_metric(method):
  import functools
  @functools.wraps(method)
  def wrapper(self, args, **kwargs):
    """ Wrapper for turning tensorflow metrics into keras metrics """
    value, update_op = method(self, args, **kwargs)
    get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update_op]):
      value = tf.identity(value)
    return value
  return wrapper


@as_keras_metric
def AUROC(y_true, y_pred, curve='ROC'):
  return tf.metrics.auc(y_true, y_pred, curve=curve)


class RocAucMetric(Callback):
  def on_train_begin(self, logs={}):
    # By default, self.params['metrics'] contains loss and the metric assigned in `model.compile()`
    if not 'val_roc_auc' in self.params['metrics']:
      self.params['metrics'].append('val_roc_auc')
    logs['val_roc_auc'] = float('-inf')

  def on_epoch_end(self, epoch, logs=None):
    if epoch % 10 == 0:
      x_test = self.validation_data[0]
      print(x_test)
      y_test = self.validation_data[1]
      predictions = self.model.predict_on_batch(x_test)
      score = roc_auc_score(y_test, predictions)
      logs['val_roc_auc'] = score


def main():
  args = parse_args()
  if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
  if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.9
  set_session(tf.Session(config=config))

  train_file_list = glob.glob(os.path.join(args.train_data, "*"))
  test_file_list = glob.glob(os.path.join(args.test_data, "*"))

  sparse_features, varlen_features, linear_sparse_features, linear_varlen_features = get_feature_info(args.feat_config)

  train_dataset = get_dataset(train_file_list, example_parser, linear_sparse_features, linear_varlen_features, args.data_type, batch_size=args.batch_size)
  test_dataset = get_dataset(test_file_list, example_parser, linear_sparse_features, linear_varlen_features, args.data_type, batch_size=args.batch_size)

  # 2.count #unique features for each sparse field and generate feature config for sequence feature
  fixlen_feature_columns = [SparseFeat(feat, feat_info[1]) for feat, feat_info in sparse_features.items()]
  linear_fixlen_feature_columns = [SparseFeat(feat, feat_info[1]) for feat, feat_info in linear_sparse_features.items()]
  varlen_feature_columns = [VarLenSparseFeat(feat, feat_info[1], feat_info[2], 'mean')
                            for feat, feat_info in varlen_features.items()]  # Notice : value 0 is for padding for sequence input feature
  linear_varlen_feature_columns = [VarLenSparseFeat(feat, feat_info[1], feat_info[2], 'mean')
                            for feat, feat_info in linear_varlen_features.items()]  # Notice : value 0 is for padding for sequence input feature
  linear_feature_columns = linear_fixlen_feature_columns + linear_varlen_feature_columns
  dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

  # 4.Define Model,compile and train
  # model = PNN(dnn_feature_columns, use_inner=True, use_outter=True, task='binary')
  # model = FNN(linear_feature_columns, dnn_feature_columns, task='binary')
  # model = DCN(linear_feature_columns, dnn_feature_columns, task='binary')
  model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[], use_fm=False, task='binary')

  trainable_count = int(np.sum([count_params(p) for p in set(model.trainable_weights)]))
  non_trainable_count = int(np.sum([count_params(p) for p in set(model.non_trainable_weights)]))
  print('Total params: {:,}'.format(trainable_count + non_trainable_count))
  print('Trainable params: {:,}'.format(trainable_count))
  print('Non-trainable params: {:,}'.format(non_trainable_count))

  # model.summary()
  model.compile("sgd", "binary_crossentropy", metrics=["accuracy", AUROC], )
  checkpoint = ModelCheckpoint(filepath=os.path.join(args.model_dir, 'model-{epoch:02d}-{loss:05f}.h5'), monitor='loss',
                               verbose=1, save_best_only=True, mode='min')
  tensorboard = TensorBoard(log_dir=args.log_dir, histogram_freq=0, write_graph=True, write_images=True)

  history = model.fit(train_dataset,
                      steps_per_epoch=args.train_size // (args.batch_size * args.num_epoch),
                      epochs=args.num_epoch,
                      verbose=1,
                      callbacks=[checkpoint, tensorboard])


if __name__ == '__main__':
  main()
