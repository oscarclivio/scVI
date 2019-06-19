#!/usr/bin/env bash


#python ../autotune.py --mode zinb --dataset retina --nb_genes 1200 --max_evals 50 --use_batches False
#python ../autotune.py --mode nb --dataset retina --nb_genes 1200 --max_evals 50 --use_batches False
python ../model_eval.py --dataset retina --n_experiments 50 --nb_genes 1200 --use_batches False \
    --infer_params_metric False \
    --zinb_hyperparams_json retina_zinb_1200_50_results.json \
    --nb_hyperparams_json retina_nb_1200_50_results.json

#python ../autotune.py --mode zinb --dataset pbmc --nb_genes 1200 --max_evals 50
#python ../autotune.py --mode nb --dataset pbmc --nb_genes 1200 --max_evals 50
python ../model_eval.py --dataset pbmc --n_experiments 50 --nb_genes 1200  \
    --infer_params_metric False \
    --zinb_hyperparams_json pbmc_zinb_1200_50_results.json \
    --nb_hyperparams_json pbmc_nb_1200_50_results.json
    

