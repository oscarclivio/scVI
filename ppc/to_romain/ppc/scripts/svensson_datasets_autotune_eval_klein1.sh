#!/usr/bin/env bash

# Klein params

# Klein params params 3
python ../autotune_ppc.py --mode zinb --dataset klein_dataset --nb_genes 1200 --max_evals 200
    
python ../autotune_ppc.py --mode nb --dataset klein_dataset --nb_genes 1200 --max_evals 200
    

python ../model_eval.py --dataset klein_dataset --n_experiments 100 \
    --zinb_hyperparams_json klein_dataset_zinb_1200_200_results.json \
    --nb_hyperparams_json klein_dataset_nb_1200_200_results.json
mv klein_dataset klein_dataset_hyperopt3_simu1
python ../model_eval.py --dataset klein_dataset --n_experiments 100 \
    --zinb_hyperparams_json klein_dataset_zinb_1200_200_results.json \
    --nb_hyperparams_json klein_dataset_nb_1200_200_results.json
mv klein_dataset klein_dataset_hyperopt3_simu2

mv klein_dataset_zinb_1200_200_results.json \
    klein_dataset_zinb_1200_200_results_hyperopt3.json
mv klein_dataset_nb_1200_200_results.json \
    klein_dataset_nb_1200_200_results_hyperopt3.json
mv trials_klein_dataset_zinb_1200_200_results \
    trials_klein_dataset_zinb_1200_200_results_hyperopt3
mv trials_klein_dataset_nb_1200_200_results \
    trials_klein_dataset_nb_1200_200_results_hyperopt3

# Klein params params 4
python ../autotune_ppc.py --mode zinb --dataset klein_dataset --nb_genes 1200 --max_evals 200
    
python ../autotune_ppc.py --mode nb --dataset klein_dataset --nb_genes 1200 --max_evals 200

python ../model_eval.py --dataset klein_dataset --n_experiments 100 \
    --zinb_hyperparams_json klein_dataset_zinb_1200_200_results.json \
    --nb_hyperparams_json klein_dataset_nb_1200_200_results.json
mv klein_dataset klein_dataset_hyperopt4_simu1

python ../model_eval.py --dataset klein_dataset --n_experiments 100 \
    --zinb_hyperparams_json klein_dataset_zinb_1200_200_results.json \
    --nb_hyperparams_json klein_dataset_nb_1200_200_results.json
mv klein_dataset klein_dataset_hyperopt4_simu2

mv klein_dataset_zinb_1200_200_results.json \
    klein_dataset_zinb_1200_200_results_hyperopt4.json
mv klein_dataset_nb_1200_200_results.json \
    klein_dataset_nb_1200_200_results_hyperopt4.json
mv trials_klein_dataset_zinb_1200_200_results \
    trials_klein_dataset_zinb_1200_200_results_hyperopt4
mv trials_klein_dataset_nb_1200_200_results \
    trials_klein_dataset_nb_1200_200_results_hyperopt4





