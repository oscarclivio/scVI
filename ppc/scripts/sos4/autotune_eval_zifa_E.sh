#!/usr/bin/env bash

# 0
#python ../autotune_colonized.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 0.0
#python ../autotune_colonized.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 0.0
#python ../model_eval_colonized.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 50 \
#    --zifa_coef 0.08 --zifa_lambda 0.0 \
#    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.0_zinb_1200_50_results.json \
#    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.0_nb_1200_50_results.json


# 1000
#python ../autotune_colonized.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 1000.0
#python ../autotune_colonized.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 1000.0
python ../../model_eval_colonized_seed.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 50 \
    --zifa_coef 0.08 --zifa_lambda 1000.0 --seed 8 \
    --zinb_hyperparams_json ../log_poisson_nb_dataset_12000_zinb_1200_50_results.json \
    --nb_hyperparams_json ../log_poisson_nb_dataset_12000_nb_1200_50_results.json

# 1e-1
#python ../autotune_colonized.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.15 --zifa_lambda 0.1
#python ../autotune_colonized.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.15 --zifa_lambda 0.1
#python ../model_eval_colonized.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 50 \
#    --zifa_coef 0.15 --zifa_lambda 0.1 \
#    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.15_0.1_zinb_1200_50_results.json \
#    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.15_0.1_nb_1200_50_results.json
