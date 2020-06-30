#!/usr/bin/env bash
rm -rf ../log/log_dsp_ctr_dataset_deepfm_linear
CUDA_VISIBLE_DEVICES="3" /home/gaoyefei/miniconda3/envs/deepctr/bin/python run_dsp_ctr_dataset.py \
  --net PNN \
  --data-type textline \
  --train-data ../data/train \
  --test-data ../data/test \
  --batch-size 1024 \
  --num-epoch 100 \
  --train-size 886881200 \
  --test-size 0 \
  --model-dir ../model \
  --log-dir ../log \
  --feat-config ../bin/field_feature.txt \
  --padding-size 5 \
  > ../log/log_dsp_ctr_dataset_deepfm_linear 2>&1 &

#  --train-size 221102858 \
#  --test-size 2233373 \