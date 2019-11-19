#!/usr/bin/env bash
rm -rf ../log/log_dsp_ctr_dataset_deepfm
CUDA_VISIBLE_DEVICES="2" /home/gaoyefei/miniconda3/envs/deepctr/bin/python run_dsp_ctr_dataset_test.py \
  --net PNN \
  --data-type textline \
  --train-data ../data/dsp_ctr/train \
  --test-data ../data/dsp_ctr/test \
  --batch-size 1024 \
  --num-epoch 1 \
  --train-size 221102858 \
  --test-size 2233373 \
  --model-dir ../model \
  --log-dir ../log \
  --feat-config ../bin/field_feature.txt \
  --padding-size 5 \
  > ../log/log_dsp_ctr_dataset_deepfm 2>&1 &