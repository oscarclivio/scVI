#!/usr/bin/env bash

# 1e-3
#python ../autotune_ppc.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
#    --zifa_coef 0.08 --zifa_lambda 0.001
#python ../autotune_ppc.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
#    --zifa_coef 0.08 --zifa_lambda 0.001
python ../model_eval.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 0.001 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.001_zinb_1200_100_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.001_nb_1200_100_results.json

# 1e-1
python ../autotune_ppc.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
    --zifa_coef 0.08 --zifa_lambda 0.1
python ../autotune_ppc.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
    --zifa_coef 0.08 --zifa_lambda 0.1
python ../model_eval.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 0.1 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.1_zinb_1200_100_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_0.1_nb_1200_100_results.json


# 1

# 1 params 1
python ../autotune_ppc.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0
python ../autotune_ppc.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0

python ../model_eval.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results.json
mv log_poisson_zifa_dataset_12000_0.08_1.0 log_poisson_zifa_dataset_12000_0.08_1.0_hyperopt1_simu1
python ../model_eval.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results.json
mv log_poisson_zifa_dataset_12000_0.08_1.0 log_poisson_zifa_dataset_12000_0.08_1.0_hyperopt1_simu2

mv log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results.json \
    log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results_hyperopt1.json
mv log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results.json \
    log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results_hyperopt1.json
mv trials_log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results \
    trials_log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results_hyperopt1
mv trials_log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results \
    trials_log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results_hyperopt1

# 1 params 2
python ../autotune_ppc.py --mode zinb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0
python ../autotune_ppc.py --mode nb --dataset log_poisson_zifa_dataset_12000 --nb_genes 1200 --max_evals 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0

python ../model_eval.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results.json
mv log_poisson_zifa_dataset_12000_0.08_1.0 log_poisson_zifa_dataset_12000_0.08_1.0_hyperopt2_simu1
python ../model_eval.py --dataset log_poisson_zifa_dataset_12000 --n_experiments 100 \
    --zifa_coef 0.08 --zifa_lambda 1.0 \
    --zinb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results.json \
    --nb_hyperparams_json log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results.json
mv log_poisson_zifa_dataset_12000_0.08_1.0 log_poisson_zifa_dataset_12000_0.08_1.0_hyperopt2_simu2

mv log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results.json \
    log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results_hyperopt2.json
mv log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results.json \
    log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results_hyperopt2.json
mv trials_log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results \
    trials_log_poisson_zifa_dataset_12000_0.08_1.0_zinb_1200_100_results_hyperopt2
mv trials_log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results \
    trials_log_poisson_zifa_dataset_12000_0.08_1.0_nb_1200_100_results_hyperopt2





