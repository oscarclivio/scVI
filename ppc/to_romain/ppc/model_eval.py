from scvi.inference.inference import UnsupervisedTrainer
from scvi.dataset import CortexDataset, RetinaDataset, HematoDataset, PbmcDataset, \
    BrainSmallDataset, ZISyntheticDatasetCorr, SyntheticDatasetCorr, LogPoissonDataset, ZIFALogPoissonDataset, \
    ZIFALogPoissonDataset
from scvi.dataset.synthetic import SyntheticDatasetCorrLogNormal, ZISyntheticDatasetCorrLogNormal
from scvi.models import VAE

import copy
import json
import argparse
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests
from metrics import *
from scvi.dataset.svensson import ZhengDataset, MacosDataset, KleinDataset, Sven1Dataset, Sven2Dataset, \
    ZhengDatasetRandom, MacosDatasetRandom, KleinDatasetRandom, Sven1DatasetRandom, Sven2DatasetRandom
import time
import torch
import numpy as np

class ModelEval:
    def __init__(self, model_fn, dataset, metrics):
        """

        :param model:
        :param dataset:
        :param metrics: List of tuples (Metric_Name:str, Metric Instance)
        """
        self.dataset = dataset
        self.metrics = metrics
        self.model_fn = model_fn
        self.trainer = None
        self.res_data = None

    def train(self, n_epochs, lr=1e-4, corruption=None, **kwargs):
        model = self.model_fn()
        self.trainer = UnsupervisedTrainer(model, self.dataset,
                                           train_size=0.8, frequency=1,
                                           early_stopping_kwargs = {
                                                        "early_stopping_metric": "elbo",
                                                        "save_best_state_metric": "elbo",
                                                        "patience": 15,
                                                        "threshold": 3,
                                                        "reduce_lr_on_plateau": True,
                                                        "lr_patience": 15,
                                                        "lr_factor": 0.5,
                                                    },
                                           n_epochs_kl_warmup=None,
                                           **kwargs)
        self.trainer.train(n_epochs, lr=lr)

        outputs = []
        imputation_is_in_metrics = False
        for metric in self.metrics:
            if metric.name != 'imputation_metric':
                metric.reset_trainer(self.trainer)
                res_dic = metric.compute()
                outputs.append(res_dic)
            else:
                imputation_is_in_metrics = True

        if imputation_is_in_metrics:
            self.trainer = UnsupervisedTrainer(model, self.dataset, **kwargs)
            self.trainer.corrupt_posteriors(rate=0.1, corruption=corruption)
            self.trainer.train(n_epochs, lr=lr)
            self.trainer.uncorrupt_posteriors()
            for metric in self.metrics:
                if metric.name == 'imputation_metric':
                    metric.reset_trainer(self.trainer)
                    res_dic = metric.compute()
                    outputs.append(res_dic)

        # Here outputs is a list of dictionnaries
        # We want to make sure that by merging them into a new dict
        # We do not loose keys
        res = {k: v for d in outputs for k, v in d.items()}
        assert len(res) == sum([len(di) for di in outputs])
        return res

    def multi_train(self, n_experiments, n_epochs, lr=1e-3, corruption=None, **kwargs):
        all_res = []
        for exp in range(n_experiments):
            all_res.append(self.train(n_epochs, corruption=corruption, lr=lr, **kwargs))
        self.res_data = pd.DataFrame(all_res)

    def write_csv(self, save_path):
        if self.res_data is None:
            raise AttributeError('No experiments yet. Use multitrain')
        self.res_data.to_csv(save_path, sep='\t')

    def write_pickle(self, save_path):
        if self.res_data is None:
            raise AttributeError('No experiments yet. Use multitrain')
        self.res_data.to_pickle(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_experiments', type=int, default=10)
    parser.add_argument('--nb_genes', type=int, default=1200)
    parser.add_argument('--zifa_coef', type=float, default=0.1)
    parser.add_argument('--zifa_lambda', type=float, default=0.0001)
    parser.add_argument('--use_batches', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--nb_hyperparams_json', type=str, default=None)
    parser.add_argument('--zinb_hyperparams_json', type=str, default=None)
    parser.add_argument('--nb', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--zinb', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--infer_params_metric', default=True, type=lambda x: (str(x).lower() == 'true'))


    args = parser.parse_args()

    dataset_name = args.dataset
    nb_genes = args.nb_genes
    n_experiments = args.n_experiments
    use_batches = args.use_batches
    zifa_coef = args.zifa_coef
    zifa_lambda = args.zifa_lambda
    nb = args.nb
    zinb = args.zinb
    infer_param_metrics = args.infer_params_metric


    def read_json(json_path):
        if json_path is not None:
            with open(json_path) as file:
                hyperparams_str = file.read()
                hyperparams = json.loads(hyperparams_str)
            lr = hyperparams.pop('lr')
        else:
            hyperparams = {}
            lr = 1e-4
        return hyperparams, lr


    if 'zifa' in dataset_name:
        dataset_name += '_' + str(zifa_coef) + '_' + str(zifa_lambda)


    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

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


        'zheng_dataset_random': partial(ZhengDatasetRandom, n_genes_random=100, seed=0),

        'macos_dataset_random': partial(MacosDatasetRandom, n_genes_random=100, seed=0),

        'klein_dataset_random': partial(KleinDatasetRandom, n_genes_random=100, seed=0),

        'klein_dataset_random97': partial(KleinDatasetRandom, n_genes_random=100, seed=97),

        'klein_dataset_random47': partial(KleinDatasetRandom, n_genes_random=100, seed=47),

        'sven1_dataset_random': partial(Sven1DatasetRandom, n_genes_random=100, seed=0),

        'sven2_dataset_random': partial(Sven2DatasetRandom, n_genes_random=100, seed=0),

        'sven1_dataset_random97': partial(Sven1DatasetRandom, n_genes_random=100, seed=97),

        'sven2_dataset_random97': partial(Sven2DatasetRandom, n_genes_random=100, seed=97),

        'sven1_dataset_random47': partial(Sven1DatasetRandom, n_genes_random=100, seed=47),

        'sven2_dataset_random47': partial(Sven2DatasetRandom, n_genes_random=100, seed=47),

    }


    MY_DATASET = datasets_mapper[dataset_name]()
    MY_DATASET.subsample_genes(new_n_genes=nb_genes)

    nb_hyperparams, lr_nb = read_json(args.nb_hyperparams_json)
    zinb_hyperparams, lr_zinb = read_json(args.zinb_hyperparams_json)

    print(nb_hyperparams, lr_nb)
    print(zinb_hyperparams, lr_zinb)




    MY_DATASET = datasets_mapper[dataset_name]()
    MY_DATASET.subsample_genes(new_n_genes=nb_genes)

    print(MY_DATASET.X.min(), MY_DATASET.X.mean(), MY_DATASET.X.max())
    print((MY_DATASET.X == 0.).mean())


    USE_BATCHES = use_batches 
    N_EXPERIMENTS = n_experiments
    N_EPOCHS = 150
    N_LL_MC_SAMPLES = 100

    if infer_param_metrics:
        MY_METRICS = [InferParamsOnZerosMetric(tag='zero_params', trainer=None, mask_zero=(MY_DATASET.X == 0))]
    else:
        MY_METRICS = []
    MY_METRICS += [
        GeneSpecificDropoutMetric(tag='gene_dropout', trainer=None),
        LikelihoodMetric(tag='ll', trainer=None, n_mc_samples=N_LL_MC_SAMPLES),
        GeneSpecificLikelihoodMetric(tag='gene_ll', trainer=None),
        SummaryStatsMetric(tag='t_dropout', trainer=None, stat_name='ks', phi_name='dropout'),
        SummaryStatsMetric(tag='t_cv', trainer=None, stat_name='ks', phi_name='cv'),
        SummaryStatsMetric(tag='t_ratio', trainer=None, stat_name='ks', phi_name='ratio'),
    ]


    def my_model_fn(reconstruction_loss, hyperparams: dict):
        return VAE(MY_DATASET.nb_genes, n_batch=MY_DATASET.n_batches * USE_BATCHES,
                   reconstruction_loss=reconstruction_loss, **hyperparams)

    def zinb_model():
        return my_model_fn('zinb', hyperparams=zinb_hyperparams)

    def nb_model():
        return my_model_fn('nb', hyperparams=nb_hyperparams)

    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    print("Use batches : ", USE_BATCHES)

    if nb:
        print("Working on NB")
        nb_eval = ModelEval(model_fn=nb_model, dataset=MY_DATASET, metrics=MY_METRICS)
        nb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform',
                            lr=lr_nb)
        nb_eval.write_csv(os.path.join(dataset_name, 'nb_{}.csv'.format(dataset_name)))
        nb_eval.write_pickle(os.path.join(dataset_name, 'nb_{}.p'.format(dataset_name)))

    if zinb:
        print("Working on ZINB")
        zinb_eval = ModelEval(model_fn=zinb_model, dataset=MY_DATASET, metrics=MY_METRICS)
        zinb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform',
                              lr=lr_zinb)
        zinb_eval.write_csv(os.path.join(dataset_name, 'zinb_{}.csv'.format(dataset_name)))
        zinb_eval.write_pickle(os.path.join(dataset_name, 'zinb_{}.p'.format(dataset_name)))
