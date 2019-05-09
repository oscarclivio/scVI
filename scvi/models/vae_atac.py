# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Multinomial, kl_divergence as kl

from scvi.models.log_likelihood import (
    log_zinb_positive,
    log_nb_positive,
    log_beta_bernoulli,
    log_zero_inflated_bernoulli,
    log_dirichlet_multinomial,
)
from scvi.models.modules import Encoder, DecoderSCVI, Decoder, LinearDecoderChromVAE
from scvi.models.utils import one_hot

import numpy as np

torch.backends.cudnn.benchmark = True


# VAE model
class VAE_ATAC(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log variational distribution
    :param reconstruction_loss:  One of

        * ``'multinomial'`` - Multinomial distribution
        * ``'bernoulli'`` - Bernoulli distribution
        * ``'zero_inflated_bernoulli'`` - ZI Bernoulli distribution
        * ``'beta-bernoulli'`` - Beta-Bernoulli distribution
        * ``'zi_multinomial'`` - ZI Multinomial
        * ``'dir-mult'`` - Dirichlet-Multinomial

    :param distribution: One of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic Normal distribution
        * ``'simplex'`` - NN transformation of normal to simplex

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = False,
        reconstruction_loss: str = "multinomial",
        log_alpha_prior=None,
        distribution: str = "ln"
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent_layers = 1  # not sure what this is for, no usages?
        self.distribution = distribution

        if log_alpha_prior is None:
            self.l_alpha_prior = torch.nn.Parameter(torch.randn(1))
        elif type(log_alpha_prior) is not str:
            self.l_alpha_prior = torch.tensor(log_alpha_prior)
        else:
            self.l_alpha_prior = None

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=distribution,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = LinearDecoderChromVAE(n_latent, n_input, n_cat_list=[n_batch])

    def get_latents(self, x, y=None):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):
        r""" samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[0]

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[2]

    def _reconstruction_loss(self, x, alpha, beta):
        # Reconstruction Loss
        if self.reconstruction_loss == "beta-bernoulli":
            reconst_loss = -log_beta_bernoulli(x, alpha, beta)
        elif self.reconstruction_loss == "bernoulli":
            reconst_loss = -torch.sum(
                torch.log(x * alpha + (1 - x) * (1 - alpha)), dim=1
            )
        elif self.reconstruction_loss == "zero_inflated_bernoulli":
            reconst_loss = -log_zero_inflated_bernoulli(x, alpha, beta)
        elif self.reconstruction_loss == 'multinomial':
            # reconst_loss = -Multinomial(probs=torch.t(alpha)).log_prob(x)
            reconst_loss = -Multinomial(probs=alpha).log_prob(x)
        elif self.reconstruction_loss == 'zi_multinomial':
            reconst_loss = -Multinomial(probs=alpha*(beta > 0.5).float()).log_prob(x)
        else: # dir-mult
            reconst_loss = -log_dirichlet_multinomial(x, alpha)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch = torch.log(1 + sample_batch)
        qz_m, qz_v, z = self.z_encoder(sample_batch)
        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            z = torch.softmax(z, dim=-1)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        if self.reconstruction_loss == "beta-bernoulli":
            log_alpha, log_beta = self.decoder(z, batch_index, y)
            alpha = torch.exp(log_alpha)
            beta = torch.exp(log_beta)
        elif self.reconstruction_loss in ["bernoulli", "zero_inflated_bernoulli"]:
            # beta is dropout
            alpha, beta = self.decoder(z, batch_index, y)
            alpha = torch.sigmoid(alpha)
        elif self.reconstruction_loss == "multinomial":
            alpha, beta = self.decoder(z, batch_index, y)
            alpha = torch.softmax(alpha, dim=-1)
            beta = None
        elif self.reconstruction_loss == "zi_multinomial":
            alpha, beta = self.decoder(z, batch_index, y)
            alpha = torch.softmax(alpha, dim=-1)
            beta = torch.sigmoid(beta)
        # dir-mult
        else:
            alpha, beta = self.decoder(z, batch_index, y)
            alpha = torch.exp(alpha)
            beta = None

        return (qz_m, qz_v, z, ql_m, ql_v, library, alpha, beta)

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution

        qz_m, qz_v, z, ql_m, ql_v, library, alpha, beta = self.inference(
            x, batch_index, y
        )

        # KL Divergence
        ap = self.l_alpha_prior
        if ap is None:
            mean = torch.zeros_like(qz_m)
            scale = torch.ones_like(qz_v)
        else:
            mean = ap - (1 / self.n_latent) * (self.n_latent * ap)
            scale = torch.sqrt(
                (1 / torch.exp(ap)) * (1 - 2 / self.n_latent)
                + (1 / self.n_latent ** 2) * (self.n_latent * 1 / torch.exp(ap))
            )

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_divergence = kl_divergence_z

        reconst_loss = self._reconstruction_loss(x, alpha, beta)

        return reconst_loss, kl_divergence
