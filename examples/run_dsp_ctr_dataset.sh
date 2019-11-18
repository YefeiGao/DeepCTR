#!/usr/bin/env bash
rm -rf ../log/log_dsp_ctr_dataset_deepfm
CUDA_VISIBLE_DEVICES="0" /home/gaoyefei/miniconda3/envs/deepctr/bin/python run_dsp_ctr_dataset.py > ../log/log_dsp_ctr_dataset_deepfm 2>&1 &