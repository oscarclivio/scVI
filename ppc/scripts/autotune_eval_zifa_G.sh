#!/usr/bin/env bash

# 0
#python ../autotune_colonized.py --mode zinb --dataset log_poisson_zifa_dataset_12$
#    --zifa_coef 0.08 --zifa_lambda 0.1
#python ../autotune_colonized.py --mode nb --dataset log_poisson_zifa_dataset_1200$
#    --zifa_coef 0.08 --zifa_lambda 0.1
python ../model_eval_colonized.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 0.0 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.0_zinb_1200_50_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.0_nb_1200_50_results.json


# 1e-1
#python ../autotune_colonized.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 0.1
#python ../autotune_colonized.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 0.1
python ../model_eval_colonized.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 0.1 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.1_zinb_1200_50_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.1_nb_1200_50_results.json


# 1e-2
#python ../autotune_colonized.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 0.01
#python ../autotune_colonized.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 50 \
#    --zifa_coef 0.08 --zifa_lambda 0.01
python ../model_eval_colonized.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 0.01 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.01_zinb_1200_50_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.01_nb_1200_50_results.json

