from scvi.inference.inference import UnsupervisedTrainer
from scvi.dataset import CortexDataset, RetinaDataset, HematoDataset, PbmcDataset, \
    BrainSmallDataset, ZISyntheticDatasetCorr, SyntheticDatasetCorr, LogPoissonDataset, ZIFALogPoissonDataset
from scvi.dataset.synthetic import SyntheticDatasetCorrLogNormal, ZISyntheticDatasetCorrLogNormal
from scvi.models import VAE

import copy
import json
import argparse
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests
from metrics import *
from zifa_full import VAE as VAE_zifa_full
# from synthetic_data import NBDataset, ZINBDataset, Mixed25Dataset, Mixed50Dataset, Mixed75Dataset


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
                                           early_stopping_kwargs={
                                               'early_stopping_metric': 'll',
                                               # 'save_best_state_metric': 'll',
                                               'patience': 15,
                                               'threshold': 3},
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
    parser.add_argument('--n_experiments', type=int, default=10)
    parser.add_argument('--nb_genes', type=int, default=1200)
    parser.add_argument('--use_batches', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--nb', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--zinb', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--nb_hyperparams_json', type=str, default=None)
    parser.add_argument('--zinb_hyperparams_json', type=str, default=None)
    parser.add_argument('--infer_params_metric', default=True, type=lambda x: (str(x).lower() == 'true'))


    args = parser.parse_args()

    infer_params_metric = args.infer_params_metric
    nb = args.nb
    zinb = args.zinb
    dataset_name = args.dataset
    nb_genes = args.nb_genes
    n_experiments = args.n_experiments
    use_batches = args.use_batches


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


    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    nb_hyperparams, kl_nb, lr_nb = read_json(args.nb_hyperparams_json)
    zinb_hyperparams, kl_zinb, lr_zinb = read_json(args.zinb_hyperparams_json)

    print(nb_hyperparams, kl_nb, lr_nb)
    print(zinb_hyperparams, kl_zinb, lr_zinb)

    datasets_mapper = {
        'pbmc': PbmcDataset,
        'cortex': CortexDataset,
        'retina': RetinaDataset,
        'hemato': HematoDataset,
        'brain_small': BrainSmallDataset,

        'corr_nb_dataset_800': partial(SyntheticDatasetCorr, lam_0=180, n_cells_cluster=800,
                                           weight_high=6, weight_low=3.5, n_overlap=0,
                                           n_genes_high=15, n_clusters=3),

        'corr_nb_dataset_2000': partial(SyntheticDatasetCorr, lam_0=180, n_cells_cluster=2000,
                                       weight_high=6, weight_low=3.5, n_overlap=0,
                                       n_genes_high=15, n_clusters=3),

        'log_poisson_nb_dataset_6000': partial(LogPoissonDataset, n_cells=6000),

        'log_poisson_zinb_dataset_1000': partial(ZIFALogPoissonDataset, n_cells=1000),

        'log_poisson_zinb_dataset_6000': partial(ZIFALogPoissonDataset, n_cells=6000),

        'log_poisson_zinb_dataset_8000': partial(ZIFALogPoissonDataset, n_cells=8000),

        'log_poisson_nb_dataset_8000': partial(LogPoissonDataset, n_cells=8000),

        'log_poisson_nb_dataset_10000': partial(LogPoissonDataset, n_cells=10000),

        'corr_nb_dataset_2000_log_normal': partial(SyntheticDatasetCorrLogNormal, library_mu=np.log(250),
                                                   n_cells_cluster=2000,
                                                   weight_high=3, weight_low=1, n_overlap=0,
                                                   n_genes_high=15, n_clusters=3),

        'corr_zinb_dataset': partial(ZISyntheticDatasetCorr, lam_0=180, n_cells_cluster=2000,
                                             weight_high=6, weight_low=3.5, n_overlap=0,
                                             n_genes_high=15, n_clusters=3,
                                             dropout_coef_high=0.05, dropout_coef_low=0.08,
                                             lam_dropout_high=0., lam_dropout_low=0.),

    }

    MY_DATASET = datasets_mapper[dataset_name]()
    MY_DATASET.subsample_genes(new_n_genes=nb_genes)

    USE_BATCHES = use_batches 
    N_EXPERIMENTS = n_experiments
    N_EPOCHS = 150
    N_LL_MC_SAMPLES = 100



    if infer_params_metric:
        MY_METRICS = [InferParamsOnZerosMetric(tag='zero_params', trainer=None, mask_zero=(MY_DATASET.X == 0))]
    else:
        MY_METRICS = []
    MY_METRICS += [
        GeneSpecificDropoutMetric(tag='gene_dropout', trainer=None),
        LikelihoodMetric(tag='ll', trainer=None, n_mc_samples=N_LL_MC_SAMPLES),
        GeneSpecificLikelihoodMetric(tag='gene_ll', trainer=None),
        ImputationMetric(tag='imputation', trainer=None),
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

    print("Use batches : ", USE_BATCHES)

    if nb :
        print("Working on NB")
        nb_eval = ModelEval(model_fn=nb_model, dataset=MY_DATASET, metrics=MY_METRICS)
        nb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform',
                            lr=lr_nb, kl=kl_nb)
        nb_eval.write_csv(os.path.join(dataset_name, 'nb_{}.csv'.format(dataset_name)))
        nb_eval.write_pickle(os.path.join(dataset_name, 'nb_{}.p'.format(dataset_name)))

    if zinb:
        print("Working on ZINB")
        zinb_eval = ModelEval(model_fn=zinb_model, dataset=MY_DATASET, metrics=MY_METRICS)
        zinb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform',
                              lr=lr_zinb, kl=kl_zinb)
        zinb_eval.write_csv(os.path.join(dataset_name, 'zinb_{}.csv'.format(dataset_name)))
        zinb_eval.write_pickle(os.path.join(dataset_name, 'zinb_{}.p'.format(dataset_name)))
