#!/usr/bin/env bash

python ../autotune_ppc.py --mode zinb --dataset klein_dataset_random --nb_genes 1200 --max_evals 100
python ../autotune_ppc.py --mode nb --dataset klein_dataset_random --nb_genes 1200 --max_evals 100
python ../model_eval.py --dataset klein_dataset_random --n_experiments 100 --nb_genes 1200  \
    --zinb_hyperparams_json klein_dataset_random_zinb_1200_100_results.json \
    --nb_hyperparams_json klein_dataset_random_nb_1200_100_results.json



python ../autotune_ppc.py --mode zinb --dataset zheng_dataset_random --nb_genes 1200 --max_evals 100
python ../autotune_ppc.py --mode nb --dataset zheng_dataset_random --nb_genes 1200 --max_evals 100
python ../model_eval.py --dataset zheng_dataset_random --n_experiments 100 --nb_genes 1200 \
    --zinb_hyperparams_json zheng_dataset_random_zinb_1200_100_results.json \
    --nb_hyperparams_json zheng_dataset_random_nb_1200_100_results.json






    

