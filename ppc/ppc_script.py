import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.stats.weightstats import ttest_ind
from tqdm import tqdm

from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE
from scvi.models.log_likelihood import compute_marginal_log_likelihood
from ppc_utils import datasets_mapper, phi_mapper, compute_ks_ppc

sns.set()

# argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--phi_name', type=str)
parser.add_argument('--dataset_name', type=str)
args = parser.parse_args()
dataset_name = args.dataset_name
phi_name = args.phi_name

# set hyperparameters
save_path = '/home/pierre/scVI/results/ppc_figs_bigrun'
logs_path = '/home/pierre/scVI/results/ppc_logs_bigrun'
n_epochs = 120
lr = 0.0004
use_batches = False
use_cuda = True
verbose = False
infer_batch_size = 32
early_stopping_kwargs = {"early_stopping_metric": "ll"}
n_samples_posterior_density = 25  # 250 is scVI-reproducibility
nb_simu = 20


my_dataset = datasets_mapper[dataset_name]()
my_dataset.subsample_genes(new_n_genes=550)
dataset_name = dataset_name
phi = phi_mapper[phi_name]


def experiment():
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
    phi_real_cell = phi(x01, axis=1)
    phi_real_gene = phi(x01, axis=0)
    phi_zi_gen_cell = phi(x_zi_gen, axis=1)
    phi_nb_gen_cell = phi(x_nb_gen, axis=1)
    phi_zi_gen_gene = phi(x_zi_gen, axis=0)
    phi_nb_gen_gene = phi(x_nb_gen, axis=0)

    # Compute KS
    my_ks_zi_cell = compute_ks_ppc(phi_real_cell, phi_zi_gen_cell)
    my_ks_zi_gene = compute_ks_ppc(phi_real_gene, phi_zi_gen_gene)
    my_ks_nb_cell = compute_ks_ppc(phi_real_cell, phi_nb_gen_cell)
    my_ks_nb_gene = compute_ks_ppc(phi_real_gene, phi_nb_gen_gene)
    return my_ks_zi_cell.mean(), my_ks_zi_gene.mean(), my_ks_nb_cell.mean(), my_ks_nb_gene.mean()


# def marginal_ll(self, verbose=False, n_mc_samples=1000):
#     ll = compute_marginal_log_likelihood(self.model, self, n_mc_samples)
#     if verbose:
#         print("True LL : %.4f" % ll)
#     return ll


def experiment_ll():
    # TODO COMPLETE
    # # Train models
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


    return


if __name__ == '__main__':
    ks_zi_cell_all, ks_zi_gene_all, ks_nb_cell_all, ks_nb_gene_all = [], [], [], []

    for training in tqdm(range(nb_simu)):
        ks_zi_cell, ks_zi_gene, ks_nb_cell, ks_nb_gene = experiment()
        ks_zi_cell_all.append(ks_zi_cell)
        ks_zi_gene_all.append(ks_zi_gene)
        ks_nb_cell_all.append(ks_nb_cell)
        ks_nb_gene_all.append(ks_nb_gene)
    ks_zi_cell_all = np.array(ks_zi_cell_all)
    ks_zi_gene_all = np.array(ks_zi_gene_all)
    ks_nb_cell_all = np.array(ks_nb_cell_all)
    ks_nb_gene_all = np.array(ks_nb_gene_all)

    _, pval_gene, _ = ttest_ind(ks_zi_gene_all, ks_nb_gene_all, alternative='smaller',
                                usevar='unequal')

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(8, 5))
    plt.sca(axes[0])
    sns.boxplot(x=ks_zi_gene_all, color='cyan')
    plt.title('Synthetic ZINB ks: {0:.6g} +- {1:.6g}'.format(ks_zi_gene_all.mean(),
                                                             ks_zi_gene_all.std()))
    plt.sca(axes[1])
    sns.boxplot(x=ks_nb_gene_all, color='salmon')
    plt.title('Synthetic NB ks: {0:.6g} +- {1:.6g}'.format(ks_nb_gene_all.mean(),
                                                           ks_nb_gene_all.std()))

    fig.suptitle('{} - KS distribution {} - Fixed gene avg of {} trainings: pvalue: {}'.format(
        dataset_name, phi_name, nb_simu, pval_gene), fontsize=12)
    plt.savefig(os.path.join(save_path,
                             '{}_{}_avg{}_ks_gene.png'.format(dataset_name, phi_name, nb_simu)))

    _, pval_cell, _ = ttest_ind(ks_zi_cell_all, ks_nb_cell_all, alternative='smaller',
                                usevar='unequal')

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(8, 5))
    plt.sca(axes[0])
    sns.boxplot(x=ks_zi_cell_all, color='cyan')
    plt.title('Synthetic ZINB ks: {0:.6g} +- {1:.6g}'.format(ks_zi_cell_all.mean(),
                                                             ks_zi_cell_all.std()))

    plt.sca(axes[1])
    sns.boxplot(x=ks_nb_cell_all, color='salmon')
    plt.title('Synthetic NB ks: {0:.6g} +- {1:.6g}'.format(ks_nb_cell_all.mean(),
                                                           ks_nb_cell_all.std()))

    fig.suptitle('{} - KS distribution {} - Fixed cell avg of {} trainings: pvalue: {}'.format(
        dataset_name, phi_name, nb_simu, pval_cell), fontsize=12)
    plt.savefig(os.path.join(save_path,
                             '{}_{}_avg{}_ks_cell.png'.format(dataset_name, phi_name, nb_simu)))

    cell_logs = [dataset_name, phi_name, 'cell', ks_zi_cell_all.mean(),
                 ks_zi_cell_all.std(), ks_nb_cell_all.mean(), ks_nb_cell_all.std(),
                 pval_cell]
    cell_logs = [str(v) for v in cell_logs]

    gene_logs = [dataset_name, phi_name, 'gene', ks_zi_gene_all.mean(),
                 ks_zi_gene_all.std(), ks_nb_gene_all.mean(),
                 ks_nb_gene_all.std(), pval_gene]
    gene_logs = [str(v) for v in gene_logs]

    text_file = open(os.path.join(logs_path, '{}_{}_gene.csv'.format(dataset_name, phi_name)), 'w')
    text_file.write(','.join(gene_logs))
    text_file.close()

    text_file = open(os.path.join(logs_path, '{}_{}_cell.csv'.format(dataset_name, phi_name)), 'w')
    text_file.write(','.join(cell_logs))
    text_file.close()
