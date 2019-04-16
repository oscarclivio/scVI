import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.stats.weightstats import ttest_ind
from tqdm import tqdm

import torch

from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE
from scvi.models.log_likelihood import compute_marginal_log_likelihood
from scipy.stats import wilcoxon
import sys

sys.path.append('/home/pierre/scVI/ppc')
from ppc_utils import datasets_mapper, phi_mapper, compute_ks_ppc

from metrics import *

sns.set()

# argparse
dataset_name = 'cortex'
phi_name = 'dropout_sum'

# set hyperparameters
save_path = '/home/oscar/scVI/results/ppc_figs_bigrun_friday'
logs_path = '/home/oscar/scVI/results/ppc_logs_bigrun_friday'
n_epochs = 120
lr = 0.0004
use_batches = False
use_cuda = True
verbose = False
infer_batch_size = 32
# early_stopping_kwargs = {"early_stopping_metric": "ll"}
early_stopping_kwargs = {}
n_samples_posterior_density = 100  # 250 is scVI-reproducibility
nb_simu = 25

my_dataset = datasets_mapper[dataset_name]()
my_dataset.subsample_genes(new_n_genes=750)
dataset_name = dataset_name
phi = phi_mapper[phi_name]

nb_cells, nb_genes = my_dataset.X.shape



zi_vae = VAE(my_dataset.nb_genes, n_batch=my_dataset.n_batches * use_batches, dropout_rate=0.2,
                 reconstruction_loss='zinb')
my_zi_trainer = UnsupervisedTrainer(zi_vae,
                                    my_dataset,
                                    train_size=0.8,
                                    use_cuda=use_cuda,
                                    kl=1, verbose=verbose, frequency=50,
                                    early_stopping_kwargs=early_stopping_kwargs)

my_zi_trainer.corrupt_posteriors(rate=0.1, corruption="uniform")
my_zi_trainer.train(n_epochs=n_epochs, lr=lr, eps=0.01)
my_zi_trainer.uncorrupt_posteriors()


imputation_metric = ImputationMetric(trainer=my_zi_trainer)
print(imputation_metric.compute())