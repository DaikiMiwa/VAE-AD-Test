#!/bin/bash

for SIGNAL in 1.0 2.0 3.0 4.0
do
  python experiment.py \
    --model_path trained_models/test_vae_model_16.h5 \
    --size 16 \
    --signal $SIGNAL \
    --thr 1.2 \
    --kernel_size 3 \
    --number_of_worker 50 \
    --number_of_iter 1000 \
    --parametric_mode decision \
    --smoothing MEAN \
    --tag_name submission_tpr_cov \
    --rho 0.25
done
