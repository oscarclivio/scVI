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

import pickle

class CovMatEstimator(object):

    def __init__(self, params={}):
        self.params = params

    def compute_cov_mat(self, dataset):
        raise NotImplementedError


class EmpCovMatEstimator(CovMatEstimator):

    def compute_cov_mat(self, dataset):

        data = dataset.X
        _, self.cov_mat = compute_empirical_means_cov_mat(data)

        return self.cov_mat


class GraphicalLassoCovMatEstimator(CovMatEstimator):

    def compute_cov_mat(self, dataset):

        data = dataset.X
        _, emp_cov  = compute_empirical_means_cov_mat(data)

        self.cov_mat = graphical_lasso(emp_cov, **self.params)[0]

        return self.cov_mat


class SparseCovMatEstimator(CovMatEstimator):

    def __init__(self, params):

        self.params = {'lmb_weights': 0.1,
                       'lr': 1e-3,
                       'lr_alt_dir': 1e-1,
                       'eps': 1e-4,
                       'n_iters_max': 1000,
                       'dichotomy': True}

        self.params.update(params)

    def compute_cov_mat(self, dataset):
        data = dataset.X
        _, emp_cov = compute_empirical_means_cov_mat(data)

        weights_mat =  np.ones_like(emp_cov) - np.identity(emp_cov.shape[0])

        self.cov_mat = sparse_cov_algo.estimate_cov_matrix(emp_cov=emp_cov,
                                                           weights_mat=weights_mat,
                                                           lmb_weights=self.params['lmb_weights'],
                                                           lr=self.params['lr'],
                                                           lr_alt_dir=self.params['lr_alt_dir'],
                                                           eps=self.params['eps'],
                                                           n_iters_max=self.params['n_iters_max'],
                                                           dichotomy=self.params['dichotomy'])

        return self.cov_mat



class SCVICovMatEstimator(CovMatEstimator):

    def __init__(self, params):

        self.params = {'n_epochs': 150}
        self.params.update(params)

        assert ('trainer_file' in params)
        self.trainer = pickle.load(open(params['trainer_file'], "rb"))



    def compute_cov_mat(self, dataset):

        px_scale, px_r, px_rate, px_dropout = get_params_inference(self.trainer, dataset, posterior_type='full')
        _, self.cov_mat = compute_empirical_means_cov_mat(px_rate)

        return self.cov_mat


class SCVISparseCovMatEstimator(CovMatEstimator):

    def __init__(self, params):

        assert ('trainer_file' in params)


        self.params = {'n_epochs': 150,
                       'lmb_weights': 0.1,
                       'lr': 1e-3,
                       'lr_alt_dir': 1e-1,
                       'eps': 1e-4,
                       'n_iters_max': 1000,
                       'dichotomy': True,
                       'verbose': False}

        self.params.update(params)

        self.trainer = pickle.load(open(params['trainer_file'], "rb"))

    def compute_cov_mat(self, dataset):

        px_scale, px_r, px_rate, px_dropout = get_params_inference(self.trainer, dataset, posterior_type='full')
        _, emp_cov = compute_empirical_means_cov_mat(px_rate)

        print("scVI empirical matrix :")
        print(emp_cov)

        weights_mat = np.ones_like(emp_cov) - np.identity(emp_cov.shape[0])

        self.cov_mat = sparse_cov_algo.estimate_cov_matrix(emp_cov=emp_cov,
                                                           weights_mat=weights_mat,
                                                           lmb_weights=self.params['lmb_weights'],
                                                           lr=self.params['lr'],
                                                           lr_alt_dir=self.params['lr_alt_dir'],
                                                           eps=self.params['eps'],
                                                           n_iters_max=self.params['n_iters_max'],
                                                           dichotomy=self.params['dichotomy'],
                                                           verbose=self.params['verbose'])


        return self.cov_mat


class SCVIGraphicalLassoCovMatEstimator(CovMatEstimator):

    def __init__(self, params):

        self.params = {'n_epochs': 150, 'tol': 1e-5}
        self.params.update(params)

        assert ('trainer_file' in params)
        self.trainer = pickle.load(open(params['trainer_file'], "rb"))

        self.params_gl = {key: value for key, value in self.params.items()
                          if key != 'n_epochs' and key != 'trainer_file' and key != 'tol'}


    def compute_cov_mat(self, dataset):

        px_scale, px_r, px_rate, px_dropout = get_params_inference(self.trainer, dataset, posterior_type='full')
        _, emp_cov = compute_empirical_means_cov_mat(px_rate)

        print("scVI empirical matrix :")
        print(emp_cov)
        print(np.min(np.linalg.eig(emp_cov)[0]))

        if np.real(np.min(np.linalg.eig(emp_cov)[0])) < self.params['tol']:
            print("scVI matrix has to be adjusted")
            emp_cov = emp_cov + self.params['tol']*np.eye(emp_cov.shape[0])

        self.cov_mat = graphical_lasso(emp_cov, **self.params_gl)[0]


        return self.cov_mat
