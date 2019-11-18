#!/usr/bin/env bash
rm -rf ../log/log_dsp_get_tfrecords
rm -rf ../data/dsp_ctr/tfrecords/train/
CUDA_VISIBLE_DEVICES="0" /home/gaoyefei/miniconda3/envs/deepctr/bin/python get_tfrecord.py --input_dir ../data.bak/dsp_ctr/train/feat.20191105.sample.train/ --output_dir ../data/dsp_ctr/tfrecords/train/ --threads 10 > ../log/log_dsp_get_tfrecords 2>&1 &