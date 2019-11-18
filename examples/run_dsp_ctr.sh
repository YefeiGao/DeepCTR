#!/usr/bin/env bash
rm -rf ../log/log_dsp_ctr_deepfm
CUDA_VISIBLE_DEVICES="1" /home/gaoyefei/miniconda3/envs/deepctr/bin/python run_dsp_ctr.py > ../log/log_dsp_ctr_deepfm 2>&1 &