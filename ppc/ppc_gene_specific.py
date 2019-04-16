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


def corr_coeff(A, B, axis=0):
    sum_prod = np.sum(A * B, axis=axis)
    prod_norms = np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    return sum_prod / prod_norms


def experiment_with_sigma_dropout():
    # Train models
    zi_vae = VAE(my_dataset.nb_genes, n_batch=my_dataset.n_batches * use_batches, dropout_rate=0.2,
                 reconstruction_loss='zinb')
    my_zi_trainer = UnsupervisedTrainer(zi_vae,
                                        my_dataset,
                                        train_size=0.8,
                                        use_cuda=use_cuda,
                                        kl=1, verbose=verbose, frequency=50,
                                        early_stopping_kwargs=early_stopping_kwargs)
    my_zi_trainer.train(n_epochs=n_epochs, lr=lr, eps=0.01)

    nb_vae = VAE(my_dataset.nb_genes, n_batch=my_dataset.n_batches * use_batches, dropout_rate=0.2,
                 reconstruction_loss='nb')
    my_nb_trainer = UnsupervisedTrainer(nb_vae,
                                        my_dataset,
                                        train_size=0.8,
                                        use_cuda=use_cuda, frequency=50,
                                        kl=1,
                                        early_stopping_kwargs=early_stopping_kwargs)
    my_nb_trainer.train(n_epochs=n_epochs, lr=lr, eps=0.01)

    # Generate synthetic data
    x_zi_gen, x01 = my_zi_trainer.train_set.generate(genes=None,
                                                     n_samples=n_samples_posterior_density,
                                                     zero_inflated=True,
                                                     batch_size=infer_batch_size)
    x_zi_gen = x_zi_gen.squeeze()
    x_nb_gen, x02 = my_nb_trainer.train_set.generate(genes=None,
                                                     n_samples=n_samples_posterior_density,
                                                     zero_inflated=False,
                                                     batch_size=infer_batch_size)
    x_nb_gen = x_nb_gen.squeeze()
    # Compute phi
    phi_real_gene = phi(x01, axis=0)
    phi_zi_gen_gene = phi(x_zi_gen, axis=0)
    phi_nb_gen_gene = phi(x_nb_gen, axis=0)

    print(phi_real_gene.shape, phi_zi_gen_gene.shape)

    # Compute imputations
    full_zi = my_zi_trainer.create_posterior(my_zi_trainer.model, my_dataset,
                                             indices=np.arange(len(my_dataset)))
    full_nb = my_zi_trainer.create_posterior(my_nb_trainer.model, my_dataset,
                                             indices=np.arange(len(my_dataset)))

    # Get dropouts
    imp_scales_zi = []
    imp_scales_nb = []
    imp_dropouts_zi = []
    imp_dropouts_nb = []
    with torch.no_grad():
        for tensors in full_zi.sequential():
            sample_batch, _, _, batch_index, labels = tensors
            px_scale_zi, _, _, px_dropout_zi = my_zi_trainer.model.inference(sample_batch,
                                                                             batch_index=batch_index,
                                                                             n_samples=1)[:4]
            imp_scales_zi += [np.array(px_scale_zi.cpu())]
            imp_dropouts_zi += [np.array(px_dropout_zi.cpu())]
        for tensors in full_nb.sequential():
            sample_batch, _, _, batch_index, labels = tensors
            px_scale_nb, _, _, px_dropout_nb = my_nb_trainer.model.inference(sample_batch,
                                                                             batch_index=batch_index,
                                                                             n_samples=1)[:4]
            imp_scales_nb += [np.array(px_scale_nb.cpu())]
            imp_dropouts_nb += [np.array(px_dropout_nb.cpu())]

    imp_scales_zi = np.concatenate(imp_scales_zi)
    imp_scales_nb = np.concatenate(imp_scales_nb)
    imp_dropouts_zi = np.concatenate(imp_dropouts_zi)
    imp_dropouts_nb = np.concatenate(imp_dropouts_nb)

    diff_gene_exp_mean_zi = np.abs(my_dataset.X - imp_scales_zi)
    diff_gene_exp_mean_zi_median = np.median(diff_gene_exp_mean_zi, axis=0)
    diff_gene_exp_mean_zi_mean = np.mean(diff_gene_exp_mean_zi, axis=0)
    diff_gene_exp_mean_nb = np.abs(my_dataset.X - imp_scales_nb)
    diff_gene_exp_mean_nb_median = np.median(diff_gene_exp_mean_nb, axis=0)
    diff_gene_exp_mean_nb_mean = np.mean(diff_gene_exp_mean_nb, axis=0)

    dropout_zi_min = imp_dropouts_zi.min(axis=0)
    dropout_zi_max = imp_dropouts_zi.max(axis=0)
    dropout_zi_mean = imp_dropouts_zi.mean(axis=0)
    dropout_zi_std = imp_dropouts_zi.std(axis=0)
    dropout_nb_min = imp_dropouts_nb.min(axis=0)
    dropout_nb_max = imp_dropouts_nb.max(axis=0)
    dropout_nb_mean = imp_dropouts_nb.mean(axis=0)
    dropout_nb_std = imp_dropouts_nb.std(axis=0)

    corr_diff_gene_exp_mean_dropout_zi = corr_coeff(diff_gene_exp_mean_zi_mean, imp_dropouts_zi,
                                                    axis=0)
    corr_diff_gene_exp_mean_dropout_nb = corr_coeff(diff_gene_exp_mean_nb_mean, imp_dropouts_nb,
                                                    axis=0)
    assert len(phi_real_gene) == nb_genes
    assert len(diff_gene_exp_mean_nb_mean) == nb_genes
    assert len(corr_diff_gene_exp_mean_dropout_nb) == nb_genes
    return (phi_zi_gen_gene, phi_nb_gen_gene, phi_real_gene,
            diff_gene_exp_mean_zi_median, diff_gene_exp_mean_zi_mean,
            corr_diff_gene_exp_mean_dropout_zi,
            diff_gene_exp_mean_nb_median, diff_gene_exp_mean_nb_mean,
            corr_diff_gene_exp_mean_dropout_nb,
            dropout_zi_min, dropout_zi_max, dropout_zi_mean, dropout_zi_std,
            dropout_nb_min, dropout_nb_max, dropout_nb_mean, dropout_nb_std)


def wilcoxon_compute(ys):
    pvals = []
    zstats = []
    for gene_idx in range(nb_genes):
        z, pval = wilcoxon(ys[gene_idx, :])
        zstats.append(z)
        pvals.append(pval)
    return np.array(zstats), np.array(pvals)


ts_zi = []
ts_nb = []
diff_gene_exp_mean_zi_medians = []
diff_gene_exp_mean_zi_means = []
corrs_diff_gene_exp_mean_dropout_zi = []
diff_gene_exp_mean_nb_medians = []
diff_gene_exp_mean_nb_means = []
corrs_diff_gene_exp_mean_dropout_nb = []
dropout_zi_mins = []
dropout_zi_maxs = []
dropout_zi_means = []
dropout_zi_stds = []
dropout_nb_mins = []
dropout_nb_maxs = []
dropout_nb_means = []
dropout_nb_stds = []

for _ in tqdm(range(nb_simu)):
    (phi_zi_gen_gene, phi_nb_gen_gene, phi_real_gene,
     diff_gene_exp_mean_zi_median, diff_gene_exp_mean_zi_mean, corr_diff_gene_exp_mean_dropout_zi,
     diff_gene_exp_mean_nb_median, diff_gene_exp_mean_nb_mean, corr_diff_gene_exp_mean_dropout_nb,
     dropout_zi_min, dropout_zi_max, dropout_zi_mean, dropout_zi_std,
     dropout_nb_min, dropout_nb_max, dropout_nb_mean,
     dropout_nb_std) = experiment_with_sigma_dropout()

    # TTEST VERSION
    #     se_zi = phi_zi_gen_gene.std(axis=-1) / np.sqrt(nb_cells)
    #     se_nb = phi_nb_gen_gene.std(axis=-1) / np.sqrt(nb_cells)
    #     t_zi = (phi_zi_gen_gene.mean(axis=-1) - phi_real_gene) / se_zi
    #     t_nb = (phi_nb_gen_gene.mean(axis=-1) - phi_real_gene) / se_nb
    #     ts_zi.append(t_zi)
    #     ts_nb.append(t_nb)

    # WILCOXON VERSION
    # y_nb = phi_nb_gen_gene - phi_real_gene.reshape((-1, 1))
    # y_zi = phi_zi_gen_gene - phi_real_gene.reshape((-1, 1))
    # zstats_nb, _ = wilcoxon_compute(y_nb)
    # zstats_zi, _ = wilcoxon_compute(y_zi)
    # ts_zi.append(zstats_zi)
    # ts_nb.append(zstats_nb)

    diff_gene_exp_mean_zi_medians.append(diff_gene_exp_mean_zi_median)
    diff_gene_exp_mean_zi_means.append(diff_gene_exp_mean_zi_mean)
    corrs_diff_gene_exp_mean_dropout_zi.append(corr_diff_gene_exp_mean_dropout_zi)
    diff_gene_exp_mean_nb_medians.append(diff_gene_exp_mean_nb_median)
    diff_gene_exp_mean_nb_means.append(diff_gene_exp_mean_nb_mean)
    corrs_diff_gene_exp_mean_dropout_nb.append(corr_diff_gene_exp_mean_dropout_nb)

    dropout_zi_mins.append(dropout_zi_min)
    dropout_zi_maxs.append(dropout_zi_max)
    dropout_zi_means.append(dropout_zi_mean)
    dropout_zi_stds.append(dropout_zi_std)

    dropout_nb_mins.append(dropout_nb_min)
    dropout_nb_maxs.append(dropout_nb_max)
    dropout_nb_means.append(dropout_nb_mean)
    dropout_nb_stds.append(dropout_nb_std)

ts_zi = np.array(ts_zi)
ts_nb = np.array(ts_nb)

print(ts_zi.shape)

plt.hist(ts_zi[:, 50])


from scipy.stats import ttest_1samp

t_zi, pval_zi = ttest_1samp(ts_zi, 0.0, axis=0)
t_nb, pval_nb = ttest_1samp(ts_nb, 0.0, axis=0)

import pandas as pd

# TODO MULTITEST QUANTILE
q_alpha = 1.645

df = pd.DataFrame(np.array([t_zi, pval_zi, t_nb, pval_nb]).T,
                  columns=['T_ZI', 'Pval_ZI', 'T_NB', 'Pval_NB'], index=my_dataset.gene_names)

# TODO: change
df.loc[:, 'zinb_good'] = (-q_alpha <= df.T_ZI) & (df.T_ZI <= q_alpha)
df.loc[:, 'nb_good'] = (-q_alpha <= df.T_NB) & (df.T_NB <= q_alpha)
df.loc[:, 'zinb_good_nb_bad'] = df.zinb_good & ~(df.nb_good)
df.loc[:, 'zinb_bad_nb_good'] = ~(df.zinb_good) & df.nb_good

df.loc[:, 'T_ZI_abs'] = df.T_ZI.abs()
df.loc[:, 'T_NB_abs'] = df.T_NB.abs()
df.loc[:, 'Difference'] = df.T_ZI.abs() - df.T_NB.abs()
df.loc[:, 'Difference_abs'] = (df.T_ZI.abs() - df.T_NB.abs()).abs()
df.loc[:, 'Gene_exp_std'] = my_dataset.X.std(axis=0)
df.loc[:, 'Gene_exp_mean'] = my_dataset.X.mean(axis=0)

for simu in range(nb_simu):
    df.loc[:, 'Diff_gene_exp_mean_zi_median_' + str(simu)] = diff_gene_exp_mean_zi_medians[simu]
    df.loc[:, 'Diff_gene_exp_mean_zi_mean_' + str(simu)] = diff_gene_exp_mean_zi_means[simu]
    df.loc[:, 'Corr_diff_gene_exp_mean_dropout_zi_' + str(simu)] = \
    corrs_diff_gene_exp_mean_dropout_zi[simu]
    df.loc[:, 'Diff_gene_exp_mean_nb_median_' + str(simu)] = diff_gene_exp_mean_nb_medians[simu]
    df.loc[:, 'Diff_gene_exp_mean_nb_mean_' + str(simu)] = diff_gene_exp_mean_nb_means[simu]
    df.loc[:, 'Corr_diff_gene_exp_mean_dropout_nb_' + str(simu)] = \
    corrs_diff_gene_exp_mean_dropout_nb[simu]

    df.loc[:, 'dropout_zi_min_' + str(simu)] = dropout_zi_mins[simu]
    df.loc[:, 'dropout_zi_max_' + str(simu)] = dropout_zi_maxs[simu]
    df.loc[:, 'dropout_zi_mean' + str(simu)] = dropout_zi_means[simu]
    df.loc[:, 'dropout_zi_std' + str(simu)] = dropout_zi_stds[simu]

    df.loc[:, 'dropout_nb_min_' + str(simu)] = dropout_nb_mins[simu]
    df.loc[:, 'dropout_nb_max_' + str(simu)] = dropout_nb_maxs[simu]
    df.loc[:, 'dropout_nb_mean' + str(simu)] = dropout_nb_means[simu]
    df.loc[:, 'dropout_nb_std' + str(simu)] = dropout_nb_stds[simu]

df = df.sort_values('T_ZI', ascending=True)
# df.loc[:, 'Corr_diff_gene_exp_imputation_dropout'] =

df = df.dropna()
df.info()

df.to_csv('gene_specific_study/{}_gene_stats_completed.csv'.format(dataset_name), sep='\t')
