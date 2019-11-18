import os
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.backend import set_session

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat, get_feature_names


if __name__ == "__main__":
  num_epochs = 1
  batch_size = 1024
  max_varlen = 5
  num_train = 221102858
  num_test = 2233373
  train_dir = "../data/dsp_ctr/train/"
  test_dir = "../data/dsp_ctr/test/"

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

  def dsp_ctr_generator(data_dir, bs, mode='train'):
    for parent, _, filenames in os.walk(data_dir):
      while True:
        for filename in filenames:
          with open(os.path.join(parent, filename), "r") as f:
            model_input_label = {}
            model_input_feat = {}
            cat_features = {}
            multi_cat_features = {}
            model_input_label["prediction_layer"] = []
            for feat in feat_name_list:
              if feat in sparse_features.keys():
                cat_features[feat] = []
              else:
                multi_cat_features[feat] = []
            while len(model_input_label["prediction_layer"]) < bs:
              line = f.readline()
              line = line.strip().split("\t")
              model_input_label["prediction_layer"].append(int(line[0]))
              for feat, idx in sparse_feat_col_ids.items():
                cat_features[feat].append(np.int32(line[idx]))
              for feat, idx in varlen_feat_col_ids.items():
                multi_feats = line[idx].split(":")
                if len(multi_feats) >= max_varlen:
                  multi_feats = multi_feats[:max_varlen]
                else:
                  multi_feats.extend([0]*(max_varlen - len(multi_feats)))
                multi_cat_features[feat].append(np.array(multi_feats, np.int32))
            model_input_feat.update(cat_features)
            model_input_feat.update(multi_cat_features)
            for feat in feat_name_list:
              if feat in cat_features.keys():
                model_input_feat[feat] = np.array(cat_features[feat], np.int32)
              else:
                model_input_feat[feat] = np.array(multi_cat_features[feat], np.int32)
            model_input_label["prediction_layer"] = np.array(model_input_label["prediction_layer"], np.int32)
            yield (model_input_feat, model_input_label)


  trainGen = dsp_ctr_generator(train_dir, batch_size, mode="train")
  testGen = dsp_ctr_generator(test_dir, batch_size, mode="train")

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
  history = model.fit_generator(trainGen,
                                steps_per_epoch=num_train // batch_size,
                                validation_data=testGen,
                                validation_steps=num_test // batch_size,
                                max_queue_size=batch_size * 15,
                                workers=10,
                                use_multiprocessing=True,
                                epochs=num_epochs)
