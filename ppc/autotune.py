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


args = parser.parse_args()

dataset_name = args.dataset
mode = args.mode
nb_genes = args.nb_genes
max_evals = args.max_evals
use_batches = args.use_batches
zifa_coef = args.zifa_coef
zifa_lambda = args.zifa_lambda

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

def compute_criterion(
    space: dict,
    gene_dataset: scvi.dataset.GeneExpressionDataset,
    # early stopping
    early_stopping_metric: str = "ll",
    save_best_state_metric: str = "ll",
    n_epochs: int = 150, train_size : float = 0.8,
    patience: int = 15,
    threshold: int = 3,  # oscillating behaviour

    # misc
    use_batches: bool = False,  # ?.?
    use_cuda: bool = True,
) -> Tuple[scvi.inference.Posterior, np.ndarray]:
    """Train and return a scVI model and sample a latent space

    :param adata: sc.AnnData object non-normalized
    :param n_latent: dimension of the latent space
    :param n_epochs: number of training epochs
    :param lr: learning rate
    :param use_batches
    :param use_cuda
    :return: (scvi.Posterior, latent_space)
    """
    ## hyperopt params
    # VAE
    n_latent = space['n_latent']
    n_hidden = space['n_hidden']
    n_layers = space['n_layers']
    dropout_rate = space['dropout_rate']
    # train func
    n_epochs = n_epochs
    lr = space['lr']

    # Train a model


    vae = VAE(
        n_input=gene_dataset.nb_genes,
        n_batch=gene_dataset.n_batches * use_batches,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        reconstruction_loss=mode
    )

    trainer = UnsupervisedTrainer(
        vae,
        gene_dataset,
        train_size=train_size,
        kl=1.,
        use_cuda=use_cuda,
        # metrics_to_monitor='ll',
        frequency=1,
        early_stopping_kwargs={
            'early_stopping_metric': early_stopping_metric,
            'save_best_state_metric': save_best_state_metric,
            'patience': patience,
            'threshold': threshold,
        },
    )
    trainer.train(n_epochs=n_epochs,
                  lr=lr)

    # return criterion
    return trainer.early_stopping.best_performance


objective_func = partial(
    compute_criterion,
    **{
        'gene_dataset': gene_dataset,
        # early stopping
        'early_stopping_metric': "ll",
        'save_best_state_metric': "ll",
        'patience': 15,
        'threshold': 3,  # oscillating behaviour
        'n_epochs': 150,
        # misc
        'use_batches': use_batches,  # ?.?
        'use_cuda': True,
    }
)


lr_choices = [1e-2, 1e-3, 1e-4]
n_latent_choices = list(range(3, 31))
n_hidden_choices = [32, 64, 128, 256]
n_layers_choices = [1, 2, 3, 4, 5]

my_space = {
    # VAE
    'n_latent': hp.choice('n_latent', n_latent_choices),  # int = 5,
    'n_hidden': hp.choice('n_hidden', n_hidden_choices),  # int = 128,
    'n_layers': hp.choice('n_layers', n_layers_choices),  # int = 1,
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.9),  # float = 0.1
    # train func
    'lr': hp.choice('lr', lr_choices),  # float = 1e-3,
}
trials = Trials()
best = fmin(
    objective_func,
    space=my_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
    show_progressbar=False,
    verbose=10
)

best["n_latent"] = n_latent_choices[best["n_latent"]]
best["n_layers"] = n_layers_choices[best["n_layers"]]
best["lr"] = lr_choices[best["lr"]]
best["n_hidden"] = n_hidden_choices[best["n_hidden"]]

with open(savefile, 'w') as fp:
    json.dump(best, fp, sort_keys=True, indent=4)
