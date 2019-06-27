#!/usr/bin/env bash

python ../autotune_ggc.py --dataset log_poisson_dataset_simple_5_10_log5_0.1_0.01_8000 \
    --mode nb --vae vae --force_autotune False --max_evals 100

python ../autotune_ggc.py --dataset log_poisson_dataset_simple_5_10_log5_0.1_0.01_8000 \
    --mode zinb --vae vae --force_autotune False --max_evals 100
