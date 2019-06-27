import numpy as np
import torch


import json
import argparse
from hyperopt import fmin, tpe, hp, Trials, rand
import scvi
from scvi.dataset import CortexDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE
from typing import Tuple
from functools import partial

import datasets_ggc
from synth_data_cov import *
import sparse_cov_algo
from sklearn.covariance import graphical_lasso
import os

from cov_mat_models import *

dataset_name = 'log_poisson_dataset_simple_5_10_log5_0.1_0.01_8000'
dataset = datasets_ggc.DATASETS_GGC[dataset_name]()

if not os.path.exists(dataset_name):
    os.makedirs(dataset_name)


MODEL_NAMES = {
    'empirical': EmpCovMatEstimator,
    'graphical_lasso': GraphicalLassoCovMatEstimator,
    'sparse': SparseCovMatEstimator,
    'scvi': SCVICovMatEstimator,
    'scvi_sparse': SCVISparseCovMatEstimator,
    'scvi_graphical_lasso': SCVIGraphicalLassoCovMatEstimator,
}

MODEL_PARAMS = {
    'empirical': {
        'empirical': {},
    },
    'graphical_lasso': {
        'alpha1': {'alpha': 1.},
        'alpha0.5': {'alpha': 0.5},
        'alpha0.2': {'alpha': 0.2},
        'alpha0.1': {'alpha': 0.1},
        #'alpha0.05': {'alpha': 0.05},
        #'alpha0.02': {'alpha': 0.02},
        #'alpha0.01': {'alpha': 0.01},
    },
    'sparse': {
        'lmbweights1': {'lmb_weights': 1.},
        'lmbweights0.5': {'lmb_weights': 0.},
        'lmbweights0.2': {'lmb_weights': 0.2},
        'lmbweights0.1': {'lmb_weights': 0.1},
        'lmbweights0.05': {'lmb_weights': 0.05},
        'lmbweights0.02': {'lmb_weights': 0.02},
        'lmbweights0.01': {'lmb_weights': 0.01},
    },
    'scvi': {
        'nb': {'trainer_file': 'best_trainer_vae_nb_log_poisson_dataset_simple_5_10_log5_0.1_0.01_8000'},
        'zinb': {'trainer_file': 'best_trainer_vae_zinb_log_poisson_dataset_simple_5_10_log5_0.1_0.01_8000'}
    }
}

MODEL_PARAMS_COMPOSITE = {}
for scvi_model_params_name, scvi_model_params in MODEL_PARAMS['scvi'].items():
    for other_model_name in ['empirical','graphical_lasso','sparse']:
        for other_model_params_name, other_model_params in MODEL_PARAMS[other_model_name].items():

            model_name = 'scvi_' + other_model_name
            if model_name not in MODEL_PARAMS_COMPOSITE:
                MODEL_PARAMS_COMPOSITE[model_name] = {}

            new_dict = {}
            new_dict.update(scvi_model_params)
            new_dict.update(other_model_params)
            MODEL_PARAMS_COMPOSITE[model_name][scvi_model_params_name + '_' + other_model_params_name] = new_dict

print(MODEL_PARAMS_COMPOSITE)
MODEL_PARAMS.update(MODEL_PARAMS_COMPOSITE)




for model_name, model_class in MODEL_NAMES.items():
    for model_params_name, model_params in MODEL_PARAMS[model_name].items():

        print(model_name, model_params_name)
        print(model_params)

        model = model_class(model_params)

        estimated_cov_mat = model.compute_cov_mat(dataset)
        np.save(dataset_name + "/" + model_name + "_" + model_params_name
                + '_' + dataset_name + '.npy', estimated_cov_mat)

for ind, (mu_orig,sigma_orig) in enumerate(zip(dataset.mus,dataset.sigmas)):
    mean,sigma = compute_theoretical_means_cov_mat_log_poisson(mu_orig, sigma_orig)
    np.save(dataset_name + "/ref_mat_"+ str(ind) + "_" + dataset_name + '.npy', sigma)


