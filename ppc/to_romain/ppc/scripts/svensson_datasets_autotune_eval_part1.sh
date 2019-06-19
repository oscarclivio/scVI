#!/usr/bin/env bash

python ../autotune_ppc.py --mode zinb --dataset sven1_dataset --nb_genes 1200 --max_evals 100
python ../autotune_ppc.py --mode nb --dataset sven1_dataset --nb_genes 1200 --max_evals 100
python ../model_eval.py --dataset sven1_dataset --n_experiments 100 --nb_genes 1200  \
    --zinb_hyperparams_json sven1_dataset_zinb_1200_100_results.json \
    --nb_hyperparams_json sven1_dataset_nb_1200_100_results.json \


python ../autotune_ppc.py --mode zinb --dataset sven2_dataset --nb_genes 1200 --max_evals 100
python ../autotune_ppc.py --mode nb --dataset sven2_dataset --nb_genes 1200 --max_evals 100
python ../model_eval.py --dataset sven2_dataset --n_experiments 100 --nb_genes 1200   \
    --zinb_hyperparams_json sven2_dataset_zinb_1200_100_results.json \
    --nb_hyperparams_json sven2_dataset_nb_1200_100_results.json



    

