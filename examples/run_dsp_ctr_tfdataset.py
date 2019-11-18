import os, sys
import tensorflow as tf
import glob
sys.path.append("../")
from tensorflow.python.keras.backend import set_session
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat, get_feature_names


if __name__ == "__main__":
  input_type = "textline"
  num_epochs = 1
  batch_size = 256
  max_varlen = 5
  num_train = 221102858
  num_test = 2233373
  # train_dir = "../data/dsp_ctr/tfrecords.test/"
  train_dir = "../data/dsp_ctr/tfrecords/train/"
  test_dir = "../data/dsp_ctr/tfrecords/test/"
  train_file_list = glob.glob(os.path.join(train_dir, "*"))
  test_file_list = glob.glob(os.path.join(test_dir, "*"))

  feat_name_list = []
  feat_dtype_dict = {"label": str}
  sparse_features = {}
  varlen_features = {}
  sparse_feat_col_ids = {}
  varlen_feat_col_ids = {}
  with open("../bin/field_feature.txt", 'r') as ff:
    for line in ff.readlines():
      field_id, _, field_name, _, bucket_size, feature_class = line.strip().split("\t")
      if bucket_size == "-1":
        bucket_size = 1000
      feat_name_list.append(field_name)
      if feature_class == "multi-cat":
        feat_dtype_dict[field_name] = str
        varlen_features[field_name] = int(bucket_size)
        varlen_feat_col_ids[field_name] = int(field_id)
      else:
        feat_dtype_dict[field_name] = str
        sparse_features[field_name] = int(bucket_size)
        sparse_feat_col_ids[field_name] = int(field_id)

  features = {feat: tf.FixedLenFeature([], tf.int64) for feat in sparse_feat_col_ids.keys()}
  varlen_feats = {feat: tf.FixedLenFeature([5], tf.int64) for feat in varlen_feat_col_ids.keys()}
  features.update(varlen_feats)
  features.update({"prediction_layer": tf.FixedLenFeature([], tf.int64)})

  def parse_example(example, is_training=True):

    parsed_features = tf.parse_single_example(example, features)
    labels = {"prediction_layer": parsed_features["prediction_layer"]}
    return parsed_features, labels


  def get_dataset(files, parse_function, num_parallel_calls=10, batch_size=256):
    print('Parsing', files)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda item: parse_function(item, feat_name_list, sparse_feat_col_ids, varlen_feat_col_ids, is_training=True),
                          num_parallel_calls=num_parallel_calls)
    # dataset = dataset.shuffle(buffer_size=batch_size*10, reshuffle_each_iteration=True)
    try:
      dataset = dataset.batch(batch_size, drop_remainder=True)
    except:
      dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(batch_size * 10)
    #return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    return iterator

  train_dataset = get_dataset(train_file_list, parse_example, batch_size=batch_size)
  # test_dataset = get_dataset(test_file_list, parse_example, batch_size=batch_size)

  # 2.count #unique features for each sparse field and generate feature config for sequence feature
  fixlen_feature_columns = [SparseFeat(feat, bucket_size)
                            for feat, bucket_size in sparse_features.items()]
  varlen_feature_columns = [VarLenSparseFeat(feat, bucket_size, max_varlen, 'mean')
                            for feat, bucket_size in varlen_features.items()]  # Notice : value 0 is for padding for sequence input feature
  linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
  dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
  feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.9
  set_session(tf.Session(config=config))

  # 4.Define Model,compile and train
  model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

  model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
  history = model.fit(train_dataset,
                      steps_per_epoch=num_train // batch_size,
                      epochs=num_epochs,
                      )

