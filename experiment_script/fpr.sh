#!/bin/bash

for SIZE in 8 16 32 64
  do 
    python experiment.py \
      --model_path trained_models/test_vae_model_$SIZE.h5 \
      --size $SIZE \
      --thr 1.2 \
      --kernel_size 3 \
      --number_of_worker 50\
      --number_of_iter 1000 \
      --parametric_mode decision \
      --smoothing MEAN \
  done
