import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat, get_feature_names


def split(x):
  key_ans = x.split(':')
  for key in key_ans:
    if key not in key2index:
      # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
      key2index[key] = len(key2index) + 1
  return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
  feat_name_list = ["label"]
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

  data = pd.DataFrame(columns=feat_name_list)
  data_dir = "../data/dsp_ctr/"
  for parent, _, filenames in os.walk(data_dir):
    for filename in filenames:
      data_part = pd.read_table(os.path.join(parent, filename), names=feat_name_list, dtype=feat_dtype_dict, header=None, sep="\t")
      data = data.append(data_part, ignore_index=True)

  target = ['label']

  # 1.Label Encoding for sparse features,and process sequence features
  for feat in sparse_features.keys():
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

  # preprocess the sequence feature
  varlen_feat_list_dict = {}
  varlen_feat_len_dict = {}
  varlen_feat_max_len_dict = {}
  key2index_dict = {}

  for feat, feat_id in varlen_feat_col_ids.items():
    key2index = {}
    feat_list = list(map(split, data[feat].values))
    feat_length = np.array(list(map(len, feat_list)))
    feat_max_len = max(feat_length)
    feat_list = pad_sequences(feat_list, maxlen=feat_max_len, padding='post', )

    key2index_dict[feat] = key2index
    varlen_feat_list_dict[feat] = feat_list
    varlen_feat_len_dict[feat] = feat_length
    varlen_feat_max_len_dict[feat] = feat_max_len

  # 2.count #unique features for each sparse field and generate feature config for sequence feature
  fixlen_feature_columns = [SparseFeat(feat, bucket_size)
                            for feat, bucket_size in sparse_features.items()]
  varlen_feature_columns = [VarLenSparseFeat(feat, bucket_size, varlen_feat_max_len_dict[feat], 'mean')
                            for feat, bucket_size in varlen_features.items()]  # Notice : value 0 is for padding for sequence input feature
  linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
  dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
  feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

  # 3.generate input data for model
  model_input = {}
  for feat in feature_names:
    if feat in sparse_features.keys():
      model_input[feat] = data[feat]
    elif feat in varlen_features.keys():
      model_input[feat] = varlen_feat_list_dict[feat]
    else:
      raise TypeError("Invalid feature column!")

  # 4.Define Model,compile and train
  model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

  model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
  history = model.fit(model_input, data[target].values,
                      batch_size=256, epochs=1, verbose=2, validation_split=0.01, )
