import json
from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import torch
import time
import scipy.stats as st


from scvi.dataset import ZIFALogPoissonDataset



DATASET_NAME = 'log_poisson_zifa_dataset_12000'
ZIFA_COEF = 0.08

ZIFA_LAMBDA_VALUES = [0., 0.001, 0.01, 0.1, 1., 10.]

big_fig, big_ax = plt.subplots(nrows=2, ncols=3, figsize=(20,12))

plot_row_values = [0, 0, 0, 1, 1, 1]
plot_col_values = [0, 1, 2, 0, 1, 2]
letters = ['a', 'b', 'c', 'd', 'e', 'f']

cs = {}


@torch.no_grad()
def get_params_inference(trainer, dataset, posterior_type='full', n_samples=100):
    assert posterior_type in ['full', 'train', 'test']

    if posterior_type == 'full':
        posterior = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
    elif posterior_type == 'train':
        posterior = trainer.train_set
    elif posterior_type == 'test':
        posterior = trainer.test_set

    if trainer.model.reconstruction_loss == 'zinb':
        px_scale_list, px_rate_list, px_dropout_list = [], [], []
        for tensors in posterior:
            sample_batch, _, _, batch_index, labels = tensors
            px_scale, px_dispersion, px_rate, px_dropout = posterior.model.inference(sample_batch,
                                                                                     batch_index=batch_index,
                                                                                     y=labels,
                                                                                     n_samples=n_samples)[0:4]

            px_dropout = 1. / (1. + torch.exp(-px_dropout))

            px_scale_list.append(px_scale.cpu().numpy().mean(axis=0))
            px_rate_list.append(px_rate.cpu().numpy().mean(axis=0))
            px_dropout_list.append(px_dropout.cpu().numpy().mean(axis=0))

        return np.concatenate(px_scale_list), \
               np.concatenate(px_rate_list), np.concatenate(px_dropout_list)

    else:
        px_scale_list, px_r_list, px_rate_list, px_dropout_list = [], [], [], []
        for tensors in posterior:
            sample_batch, _, _, batch_index, labels = tensors
            px_scale, _, px_rate = posterior.model.inference(sample_batch,
                                                             batch_index=batch_index,
                                                             y=labels,
                                                             n_samples=n_samples)[0:3]

            px_scale_list.append(px_scale.cpu().numpy().mean(axis=2))
            px_rate_list.append(px_rate.cpu().numpy().mean(axis=2))

        return np.concatenate(px_scale_list), \
               np.concatenate(px_rate_list), None

for zifa_lambda, plot_row, plot_col, letter in zip(ZIFA_LAMBDA_VALUES, plot_row_values, plot_col_values, letters):

    dataset_name = DATASET_NAME
    if 'zifa' in dataset_name:
        dataset_name += '_' + str(ZIFA_COEF) + '_' + str(zifa_lambda)

    zinb_hyperparameters_json = 'scripts/' + dataset_name + '_zinb_1200_50_results.json'


    ## Get the full mask whether a zero is biological or technical

    datasets_mapper = {

        'log_poisson_zifa_dataset_12000_' + str(ZIFA_COEF) + '_' + str(zifa_lambda): \
            partial(ZIFALogPoissonDataset, n_cells=12000, dropout_coef=ZIFA_COEF, dropout_lambda=zifa_lambda),

        }


    my_dataset = datasets_mapper[dataset_name]()

    mask_zero_all = (my_dataset.X == 0)[np.newaxis, :, :]
    biological_zeros = (my_dataset.mask_zero_biological)[mask_zero_all]
    technical_zeros = (~my_dataset.mask_zero_biological & my_dataset.mask_zero_zi)[mask_zero_all]


    ## Extract dropout, rate and scale metrics in the CSVs

    # Load ZINB optimal hyperparameters and model

    def read_json(json_path):
        if json_path is not None:
            with open(json_path) as file:
                hyperparams_str = file.read()
                hyperparams = json.loads(hyperparams_str)
            kl = 1.
            lr = hyperparams.pop('lr')
        else:
            hyperparams = {}
            kl = 1
            lr = 1e-4
        return hyperparams, kl, lr

    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    zinb_hyperparams, kl_zinb, lr_zinb = read_json(zinb_hyperparameters_json)

    model = VAE(my_dataset.nb_genes, n_batch=my_dataset.n_batches,
                reconstruction_loss='zinb', **zinb_hyperparams)

    # Train model

    trainer = UnsupervisedTrainer(model, my_dataset, train_size=0.8, frequency=1,
                                  early_stopping_kwargs={
                                       'early_stopping_metric': 'll',
                                       # 'save_best_state_metric': 'll',
                                       'patience': 15,
                                       'threshold': 3},
                                  kl=kl_zinb)
    trainer.train(n_epochs=150, lr=lr_zinb)

    # Sample 100 posteriors, retrieve average dropout, scale and rate

    scales_all, rates_all, dropout_probs_all = get_params_inference(trainer,
                                                                    my_dataset,
                                                                    posterior_type='full',
                                                                    n_samples=100)

    scales_all = scales_all[(my_dataset.X == 0)]
    rates_all = rates_all[(my_dataset.X == 0)]
    dropout_probs_all = dropout_probs_all[(my_dataset.X == 0)]

    dropout_probs_biological = dropout_probs_all[biological_zeros]
    dropout_probs_technical = dropout_probs_all[technical_zeros]

    scales_biological = scales_all[biological_zeros]
    scales_technical = scales_all[technical_zeros]

    rates_biological = rates_all[biological_zeros]
    rates_technical = rates_all[technical_zeros]

    ## Plots !!

    # Function to create density plots

    CMAPS = {"tech": "Reds", "bio": "Blues"}
    LABELS = {"bio": "Limited sensitivity", "tech": "Dropout"}

    def density_plot(dropouts, means, zero_type, ax=None):

        if dropouts.size == 0:
            pass

        else:
            cmap = CMAPS[zero_type]
            color = sns.color_palette(cmap)[-2]
            label = LABELS[zero_type]

            # Get the colormap colors
            cmap = plt.cm.get_cmap(cmap)
            my_cmap = cmap(np.arange(cmap.N))
            # Set alpha
            my_cmap[:, -1] = np.linspace(0.25, 0.85, cmap.N)
            my_cmap[:int(cmap.N/10), -1] = 0.
            # Create new colormap
            my_cmap = ListedColormap(my_cmap)


            xmin = dropouts.min()
            xmax = dropouts.max()
            ymin = means.min()
            ymax = means.max()

            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([dropouts, means])
            kernel = st.gaussian_kde(values)
            fig = np.reshape(kernel(positions).T, xx.shape)

            cs[zero_type] = ax.contourf(xx, yy, fig, cmap=my_cmap)

            xmin = 0.5*(xmin + xx[fig > 0.03].min())
            xmax = 0.5*(xmax + xx[fig > 0.03].max()) + 0.2
            ymin = 0.5*(ymin + yy[fig > 0.03].min())
            ymax = 0.5*(ymax + yy[fig > 0.03].max())

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.text(x=0.99, y=0.99 - 0.03*(zero_type=='tech'), s=label, color=color,
                    horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)



    # All
    density_plot(np.log10(dropout_probs_technical), np.log10(rates_technical), 'tech', ax=big_ax[plot_row, plot_col])
    density_plot(np.log10(dropout_probs_biological), np.log10(rates_biological), 'bio', ax=big_ax[plot_row, plot_col])
    if plot_row == plot_row_values[-1]:
        big_ax[plot_row, plot_col].set_xlabel("Dropout probabilities ($\log_{10}$ scale)")
    if plot_col == 0:
        big_ax[plot_row, plot_col].set_ylabel("NB means ($\log_{10}$ scale)")
    big_ax[plot_row, plot_col].set_title("$\lambda = {val}$ ({letter})".format(val=zifa_lambda, letter=letter))

big_fig.colorbar(cs['bio'], ax=big_ax.ravel().tolist())
big_fig.colorbar(cs['tech'], ax=big_ax.ravel().tolist())

plt.savefig("results_zero_study/datasets_all_rates_dropout_probs_all_dpi450.png", dpi=450)
plt.close()
