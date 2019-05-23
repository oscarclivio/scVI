import numpy as np
import torch
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

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

from scvi.dataset import LogPoissonDatasetGeneral

class LogPoissonDatasetSimple(LogPoissonDatasetGeneral):

    def __init__(self, n_genes=50, n_cells=100, dropout=0.):

        pi = [0.]
        self.mu = torch.Tensor(np.random.uniform(1., 2., size=(n_genes)))
        mus = [self.mu]
        sigma_gt = 0.1*(np.identity(n_genes) ) #+ 1 / (10 * n_genes) * np.random.binomial(1, 0.1, (n_genes, n_genes)))
        assert (np.min(np.linalg.eig(sigma_gt)[0]) > 1e-4)
        sigma_gt = torch.Tensor(sigma_gt)
        sigmas = [sigma_gt]

        self.sigma_gt = sigma_gt

        super(LogPoissonDatasetSimple, self).__init__(pi, mus, sigmas, n_cells, dropout)


def compute_theoretical_means_cov_mat_log_poisson(mean_normal, sigma_normal):

    mean_lp = torch.exp(mean_normal + 0.5 * torch.diag(sigma_normal))

    mean_lp_unsq = mean_lp.unsqueeze(1)
    mean_lp_unsq_t = mean_lp.unsqueeze(0)
    print(torch.mm(mean_lp_unsq,mean_lp_unsq_t).size())

    sigma_lp = torch.mm(mean_lp_unsq,mean_lp_unsq_t) * (torch.exp(sigma_normal) - 1.)
    sigma_lp += torch.diag(mean_lp)

    return mean_lp, sigma_lp


def compute_empirical_means_cov_mat_log_poisson(X):

    mean_emp = np.mean(X, axis=0)
    mean_emp_unsq = mean_emp.reshape(1, mean_emp.size)
    sigma_emp = (X - mean_emp_unsq).T.dot(X - mean_emp_unsq) / (X.shape[0] - 1)

    return mean_emp, sigma_emp

def compute_corr_mat_from_cov_mat(sigma):

    vars_diag = np.diag(sigma)
    sigma_corr = vars_diag.reshape((vars_diag.shape[0],1)).dot(vars_diag.reshape((1,vars_diag.shape[0])))
    return sigma / np.sqrt(sigma_corr)


@torch.no_grad()
def get_params_inference(trainer, dataset, posterior_type='full', n_samples=1):

    assert posterior_type in ['full','train','test']

    if posterior_type == 'full':
        posterior = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
    elif posterior_type == 'train':
        posterior = trainer.train_set
    elif posterior_type == 'test':
        posterior = trainer.test_set

    if trainer.model.reconstruction_loss == 'zinb':
        px_scale_list, px_r_list, px_rate_list, px_dropout_list = [], [], [], []
        for tensors in posterior:
            sample_batch, _, _, batch_index, labels = tensors
            px_scale, px_r, px_rate, px_dropout = posterior.model.inference(sample_batch,
                                                                            batch_index=batch_index,
                                                                            y=labels,
                                                                            n_samples=n_samples)[0:4]
            px_scale_list.append(px_scale.cpu().numpy())
            px_r_list.append(px_r.cpu().numpy())
            px_rate_list.append(px_rate.cpu().numpy())
            px_dropout_list.append(px_dropout.cpu().numpy())

        return np.concatenate(px_scale_list), np.concatenate(px_r_list),\
               np.concatenate(px_rate_list), np.concatenate(px_dropout_list)

    else:
        px_scale_list, px_r_list, px_rate_list, px_dropout_list = [], [], [], []
        for tensors in posterior:
            sample_batch, _, _, batch_index, labels = tensors
            px_scale, px_r, px_rate = posterior.model.inference(sample_batch,
                                                                batch_index=batch_index,
                                                                y=labels,
                                                                n_samples=n_samples)[0:3]
            px_scale_list.append(px_scale.cpu().numpy())
            px_r_list.append(px_r.cpu().numpy())
            px_rate_list.append(px_rate.cpu().numpy())

        return np.concatenate(px_scale_list), np.concatenate(px_r_list), \
               np.concatenate(px_rate_list), None





if __name__ == '__main__':

    dataset = LogPoissonDatasetSimple(n_cells=10000)

    # Theoretical and empirical covariance matrix
    mean_normal = dataset.mu
    sigma_normal = dataset.sigma_gt

    mean_lp, sigma_lp = compute_theoretical_means_cov_mat_log_poisson(mean_normal, sigma_normal)
    mean_emp, sigma_emp = compute_empirical_means_cov_mat_log_poisson(dataset.X)

    # Train a model
    nb_model = VAE(dataset.nb_genes, n_batch=dataset.n_batches, reconstruction_loss='nb')
    trainer = UnsupervisedTrainer(nb_model, dataset)
    trainer.train(n_epochs=150)

    # Generate inference params and compute empirical covariance matrix on them (ex: rate)
    px_scale, px_r, px_rate, px_dropout = get_params_inference(trainer, dataset, posterior_type='full')
    mean_emp_rate, sigma_emp_rate = compute_empirical_means_cov_mat_log_poisson(px_rate)

    # View results
    #print(mean_normal)
    #print(sigma_normal)
    #print("=== Means ===")
    #print(mean_lp.cpu().numpy())
    #print(mean_emp)
    #print(np.abs(mean_lp.cpu().numpy() - mean_emp))
    #print(mean_emp_rate)
    #print(np.abs(mean_lp.cpu().numpy() - mean_emp_rate))


    print("=== Sigmas ===")

    print("Cov matrix - Ground truth")
    print(sigma_lp.cpu().numpy())
    print(sigma_lp.cpu().numpy().shape)


    sns.heatmap(sigma_lp.cpu().numpy())
    plt.axis('equal')
    plt.title('Cov matrix - Ground truth')
    plt.savefig("cov_mats_1_gt.png")
    plt.close()

    print("Cov matrix - Empirical")
    print(sigma_emp)
    sns.heatmap(sigma_emp)
    plt.axis('equal')
    plt.title('Cov matrix - Empirical')
    plt.savefig("cov_mats_2_emp.png")
    plt.close()

    print("Abs diff with ground truth - Empirical")
    print(np.abs(sigma_lp.cpu().numpy() - sigma_emp))

    print("Cov matrix - Poisson rates")
    print(sigma_emp_rate)

    sns.heatmap(sigma_emp_rate)
    plt.axis('equal')
    plt.title('Cov matrix - scVI-empirical')
    plt.savefig("cov_mats_3_scvi.png")
    plt.close()
    print("Abs diff with ground truth - Poisson rates")
    print(np.abs(sigma_lp.cpu().numpy() - sigma_emp_rate))

    plt.savefig('cov_mats.png')

