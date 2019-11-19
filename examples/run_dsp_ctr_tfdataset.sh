#!/usr/bin/env bash
rm -rf ../log/log_dsp_ctr_tfdataset_deepfm
CUDA_VISIBLE_DEVICES="2" /home/gaoyefei/miniconda3/envs/deepctr/bin/python run_dsp_ctr_tfdataset.py > ../log/log_dsp_ctr_tfdataset_deepfm 2>&1 &