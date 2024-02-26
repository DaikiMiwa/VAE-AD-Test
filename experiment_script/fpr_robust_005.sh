#!/bin/bash

SIZE=16

for NOISE in skewnorm exponnorm gennormsteep gennormflat t
  do
  for DISTANCE in 0.01 0.02 0.03 0.04
    do 
      python experiment.py \
        --model_path trained_models/test_vae_model_$SIZE.h5 \
        --size $SIZE \
        --thr 1.2 \
        --kernel_size 3 \
        --number_of_worker 40 \
        --number_of_iter 1000 \
        --parametric_mode decision \
        --noise_distribution $NOISE \
        --ws_distance $DISTANCE \
        --tag_name submission_fpr_robust_after
  done
done
