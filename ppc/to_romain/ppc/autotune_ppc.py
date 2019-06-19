import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

import json
import argparse
from hyperopt import fmin, tpe, hp, Trials, rand
import scvi
from scvi.dataset import CortexDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE
from typing import Tuple
from functools import partial
import numpy as np
import time
from scvi.dataset.svensson import ZhengDataset, MacosDataset, KleinDataset, Sven1Dataset, Sven2Dataset
from scvi.inference.autotune import auto_tune_scvi_model
import logging

from scvi.dataset import CortexDataset, RetinaDataset, HematoDataset, PbmcDataset, \
    BrainSmallDataset, ZIFALogPoissonDataset



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--nb_genes', type=int, default=1200)
parser.add_argument('--max_evals', type=int)
parser.add_argument('--use_batches', default=True,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--zifa_coef', type=float, default=0.1)
parser.add_argument('--zifa_lambda', type=float, default=0.0001)
parser.add_argument('--parallel', default=False,
                    type=lambda x: (str(x).lower() == 'true'))


args = parser.parse_args()

dataset_name = args.dataset
mode = args.mode
nb_genes = args.nb_genes
max_evals = args.max_evals
use_batches = args.use_batches
zifa_coef = args.zifa_coef
zifa_lambda = args.zifa_lambda
parallel = args.parallel

if 'zifa' in dataset_name:
    dataset_name += '_' + str(zifa_coef) + '_' + str(zifa_lambda)

datasets_mapper = {
    'pbmc': PbmcDataset,
    'cortex': CortexDataset,
    'retina': RetinaDataset,
    'hemato': HematoDataset,
    'brain_small': BrainSmallDataset,

    'log_poisson_zifa_dataset_12000_' + str(zifa_coef) + '_' + str(zifa_lambda): \
        partial(ZIFALogPoissonDataset, n_cells=12000, dropout_coef=zifa_coef, dropout_lambda=zifa_lambda),

    'zheng_dataset': ZhengDataset,

    'macos_dataset': MacosDataset,

    'klein_dataset': KleinDataset,

    'sven1_dataset': Sven1Dataset,

    'sven2_dataset': Sven2Dataset,

}


gene_dataset = datasets_mapper[dataset_name]()
gene_dataset.subsample_genes(new_n_genes=nb_genes)

savefile = '{}_{}_{}_{}_results.json'.format(dataset_name, mode, nb_genes, max_evals)

np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))

lr_choices = [1e-2, 1e-3, 1e-4]
n_latent_choices = list(range(3, 31))
n_hidden_choices = [32, 64, 128, 256]
n_layers_choices = [1, 2, 3, 4, 5]

my_space = {
    "model_tunable_kwargs": {
        "n_latent": hp.choice("n_latent", n_latent_choices),  # [5, 15]
        "n_hidden": hp.choice("n_hidden", n_hidden_choices),
        "n_layers": hp.choice("n_layers", n_layers_choices),
        "dropout_rate": hp.uniform('dropout_rate', 0.1, 0.9),
    },
    "train_func_tunable_kwargs": {
        "lr": hp.choice("lr", lr_choices)
    }
}


early_stopping_kwargs={'early_stopping_metric': "ll",
                       'save_best_state_metric': "ll",
                       'patience': 15,
                       'threshold': 3}

logging.getLogger('scvi.inference.autotune').setLevel(logging.DEBUG)


trials = auto_tune_scvi_model(exp_key=savefile.replace(".json", ""), gene_dataset=gene_dataset,
                              space=my_space, max_evals=max_evals,
                              model_specific_kwargs={'reconstruction_loss': mode},
                              use_batches=use_batches,
                              trainer_specific_kwargs={'kl': 1., 'early_stopping_kwargs': early_stopping_kwargs,
                                                         'use_cuda': True},
                              train_func_specific_kwargs={'n_epochs': 150},
                              train_best=False, parallel=parallel)

best = trials.argmin

best["n_latent"] = n_latent_choices[best["n_latent"]]
best["n_layers"] = n_layers_choices[best["n_layers"]]
best["lr"] = lr_choices[best["lr"]]
best["n_hidden"] = n_hidden_choices[best["n_hidden"]]

with open(savefile, 'w') as fp:
    json.dump(best, fp, sort_keys=True, indent=4)
