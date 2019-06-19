#!/usr/bin/env bash

python ../autotune_ppc.py --mode zinb --dataset cortex --nb_genes 1200 --max_evals 100
python ../autotune_ppc.py --mode nb --dataset cortex --nb_genes 1200 --max_evals 100
python ../model_eval.py --dataset cortex --n_experiments 100 --nb_genes 1200  \
    --infer_params_metric False \
    --zinb_hyperparams_json cortex_zinb_1200_100_results.json \
    --nb_hyperparams_json cortex_nb_1200_100_results.json


python ../autotune_ppc.py --mode zinb --dataset hemato --nb_genes 1200 --max_evals 100
python ../autotune_ppc.py --mode nb --dataset hemato --nb_genes 1200 --max_evals 100
python ../model_eval.py --dataset hemato --n_experiments 100 --nb_genes 1200  \
    --infer_params_metric False \
    --zinb_hyperparams_json hemato_zinb_1200_100_results.json \
    --nb_hyperparams_json hemato_nb_1200_100_results.json




python ../autotune_ppc.py --mode zinb --dataset brain_small --nb_genes 1200 --max_evals 100 --use_batches False
python ../autotune_ppc.py --mode nb --dataset brain_small --nb_genes 1200 --max_evals 100 --use_batches False
python ../model_eval.py --dataset brain_small --n_experiments 100 --nb_genes 1200 --use_batches False \
    --infer_params_metric False \
    --zinb_hyperparams_json brain_small_zinb_1200_100_results.json \
    --nb_hyperparams_json brain_small_nb_1200_100_results.json




