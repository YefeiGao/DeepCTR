import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import set_session, get_session, count_params

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat, get_feature_names


def split(x):
  key_ans = x.split('|')
  for key in key_ans:
    if key not in key2index:
      # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
      key2index[key] = len(key2index) + 1
  return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.9
  set_session(tf.Session(config=config))
  data = pd.read_csv("./movielens_sample.txt")
  sparse_features = ["movie_id", "user_id",
                     "gender", "age", "occupation", "zip", ]
  target = ['rating']

  # 1.Label Encoding for sparse features,and process sequence features
  for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
  # preprocess the sequence feature

  key2index = {}
  genres_list = list(map(split, data['genres'].values))
  genres_length = np.array(list(map(len, genres_list)))
  max_len = max(genres_length)
  # Notice : padding=`post`
  genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

  # 2.count #unique features for each sparse field and generate feature config for sequence feature

  fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                            for feat in sparse_features]
  varlen_feature_columns = [VarLenSparseFeat('genres', len(
    key2index) + 1, max_len, 'mean')]  # Notice : value 0 is for padding for sequence input feature

  linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
  dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

  feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

  # 3.generate input data for model
  model_input = {name: data[name] for name in feature_names}  #
  model_input["genres"] = genres_list

  # 4.Define Model,compile and train
  model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')

  trainable_count = int(np.sum([count_params(p) for p in set(model.trainable_weights)]))
  non_trainable_count = int(np.sum([count_params(p) for p in set(model.non_trainable_weights)]))
  print('Total params: {:,}'.format(trainable_count + non_trainable_count))
  print('Trainable params: {:,}'.format(trainable_count))
  print('Non-trainable params: {:,}'.format(non_trainable_count))

  model.compile("adam", "mse", metrics=['mse'], )
  history = model.fit(model_input, data[target].values,
                      batch_size=256, epochs=10000, verbose=2, validation_split=0.2, )
