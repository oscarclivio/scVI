#!/usr/bin/env bash

python ../model_eval.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 50 \
    --zifa_coef 0.08 --zifa_lambda 10.0 --nb False \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_10.0_zinb_1200_100_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_10.0_nb_1200_100_results.json
