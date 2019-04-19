from scvi.inference.inference import UnsupervisedTrainer
from scvi.dataset import CortexDataset, RetinaDataset, HematoDataset, PbmcDataset, \
    BrainSmallDataset
from scvi.models import VAE

import copy
import json
import argparse
import pandas as pd
from statsmodels.stats.multitest import multipletests
from metrics import *
from zifa_full import VAE as VAE_zifa_full
from zifa_half import VAE as VAE_zifa_half


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

    def train(self, n_epochs, lr=1e-3, corruption=None, **kwargs):
        model = self.model_fn()
        self.trainer = UnsupervisedTrainer(model, self.dataset,
                                           # early_stopping_kwargs={
                                           #     'early_stopping_metric': 'll',
                                           #     'save_best_state_metric': 'll',
                                           #     'patience': 15,
                                           #     'threshold': 3},
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


def statistic_metric(subdf, stats_key=None):
    ser = subdf[stats_key]
    ser = ser.mean(axis=1)


def outlier_metric(subdf, pvals_keys=None):
    def outliers(pval_col, alpha=0.05, method='fdr_bh', use_alpha_new=True):
        pval_col_corrected, alpha_new = multipletests(pval_col, alpha=alpha, method=method)[1:3]
        alpha_boundary = alpha_new if use_alpha_new else alpha
        return pval_col <= alpha_boundary

    assert len(pvals_keys) == 2
    key1, key2 = pvals_keys
    # TODO: maybe change below
    pvals1 = subdf[key1][:, 0]
    pvals2 = subdf[key2][:, 0]
    outliers1 = outliers(pvals1)
    outliers2 = outliers(pvals2)

    # TODO Verifier pas trompé
    good1bad2 = (~outliers1) * outliers2
    bad1good2 = outliers1 * (~outliers2)
    return pd.Series([good1bad2.sum(), bad1good2.sum()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--number_genes', type=int, default=1200)
    parser.add_argument('--nb_hyperparams_json', type=str)
    parser.add_argument('--zinb_hyperparams_json', type=str)
    args = parser.parse_args()

    dataset_name = args.dataset
    number_genes = args.number_genes

    with open(args.nb_hyperparams_json) as file:
        nb_hyperparams_str = file.read()
        nb_hyperparams = json.loads(nb_hyperparams_str)
        kl_nb = nb_hyperparams.pop('kl_weight')
        lr_nb = nb_hyperparams.pop('lr')

        print(nb_hyperparams)

    with open(args.zinb_hyperparams_json) as file:
        zinb_hyperparams_str = file.read()
        zinb_hyperparams = json.loads(zinb_hyperparams_str)
        kl_zinb = zinb_hyperparams.pop('kl_weight')
        lr_zinb = zinb_hyperparams.pop('lr')

    datasets_mapper = {
        'pbmc': PbmcDataset,
        'cortex': CortexDataset,
        'retina': RetinaDataset,
        'hemato': HematoDataset,
        'brain_small': BrainSmallDataset
    }

    MY_DATASET = datasets_mapper[dataset_name]()
    MY_DATASET.subsample_genes(new_n_genes=number_genes)

    USE_BATCHES = True
    N_EXPERIMENTS = 20
    N_EPOCHS = 150
    N_LL_MC_SAMPLES = 25
    MY_METRICS = [
        LikelihoodMetric(tag='ll', trainer=None, n_mc_samples=N_LL_MC_SAMPLES),
        ImputationMetric(tag='imputation', trainer=None),
        SummaryStatsMetric(tag='t_dropout', trainer=None, stat_name='ks', phi_name='dropout'),
        # SummaryStatsMetric(tag='t_cv', trainer=None, stat_name='ks', phi_name='cv'),
        # SummaryStatsMetric(tag='t_ratio', trainer=None, stat_name='ks', phi_name='ratio'),
        # ('diff', DifferentialExpressionMetric(trainer=None, )),
    ]


    def my_model_fn(reconstruction_loss, hyperparams: dict):
        return VAE(MY_DATASET.nb_genes, n_batch=MY_DATASET.n_batches * USE_BATCHES,
                   reconstruction_loss=reconstruction_loss, **hyperparams)

    def zinb_model():
        return my_model_fn('zinb', hyperparams=zinb_hyperparams)

    def nb_model():
        return my_model_fn('nb', hyperparams=nb_hyperparams)

    def zifa_half_fn():
        return VAE_zifa_half(MY_DATASET.nb_genes, n_batch=MY_DATASET.n_batches * USE_BATCHES,
                             decay_mode='gene')

    def zifa_full_fn():
        return VAE_zifa_full(MY_DATASET.nb_genes, n_batch=MY_DATASET.n_batches * USE_BATCHES,
                             decay_mode='gene')

    zinb_eval = ModelEval(model_fn=zinb_model, dataset=MY_DATASET, metrics=MY_METRICS)
    zinb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform',
                          lr=lr_zinb, kl=kl_zinb)

    nb_eval = ModelEval(model_fn=nb_model, dataset=MY_DATASET, metrics=MY_METRICS)
    nb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform',
                        lr=lr_nb, kl=kl_nb)

    zifa_half_eval = ModelEval(model_fn=zifa_half_fn, dataset=MY_DATASET, metrics=MY_METRICS)
    zifa_half_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform')

    zifa_full_eval = ModelEval(model_fn=zifa_full_fn, dataset=MY_DATASET, metrics=MY_METRICS)
    zifa_full_eval .multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform')



    # save files
    zinb_eval.write_csv('zinb_{}_hyperopt.csv'.format(dataset_name))
    nb_eval.write_csv('nb_{}_hyperopt.csv'.format(dataset_name))
    zifa_half_eval.write_csv('zifa_half_{}.csv'.format(dataset_name))
    zifa_full_eval.write_csv('zifa_full_{}.csv'.format(dataset_name))

    zinb_eval.write_pickle('zinb_{}_hyperopt.p'.format(dataset_name))
    nb_eval.write_pickle('nb_{}_hyperopt.p'.format(dataset_name))
    zifa_half_eval.write_pickle('zifa_half_{}.p'.format(dataset_name))
    zifa_full_eval.write_pickle('zifa_full_{}.p'.format(dataset_name))

    # def zifa_fn(decay_mode='gene', model='half'):
    #     """
    #     decay_mode = 'gene' or ''
    #     model = 'half' or 'full'
    #     """
    #     if model == 'half':
    #         return VAE_zifa_full(
    #             MY_DATASET.nb_genes,
    #             n_batch=MY_DATASET.n_batches * USE_BATCHES,
    #             decay_mode=decay_mode
    #         )
    #     else:
    #         return VAE_zifa_half(
    #             MY_DATASET.nb_genes,
    #             n_batch=MY_DATASET.n_batches * USE_BATCHES,
    #             decay_mode=decay_mode
    #         )
