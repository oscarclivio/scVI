from scvi.inference.inference import UnsupervisedTrainer
from scvi.dataset import CortexDataset, RetinaDataset, HematoDataset, PbmcDataset, \
    BrainSmallDataset
from scvi.models import VAE

import copy
import pandas as pd
from statsmodels.stats.multitest import multipletests
from metrics import *


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

    def train(self, n_epochs, corruption=None, **kwargs):
        model = self.model_fn()
        self.trainer = UnsupervisedTrainer(model, self.dataset, **kwargs)
        self.trainer.train(n_epochs)

        outputs = []
        imputation_is_in_metrics = False
        for metric_tag, metric in self.metrics:
            if metric.name != 'imputation_metric':
                metric.reset_trainer(self.trainer)
                res_dic = metric.compute()
                new_dic = {metric_tag + '_' + k: v for k, v in res_dic.items()}
                outputs.append(new_dic)
            else:
                imputation_is_in_metrics = True

        if imputation_is_in_metrics:
            self.trainer = UnsupervisedTrainer(model, self.dataset, **kwargs)
            self.trainer.corrupt_posteriors(rate=0.1, corruption=corruption)
            self.trainer.train(n_epochs)
            self.trainer.uncorrupt_posteriors()
            for metric_tag, metric in self.metrics:
                if metric.name == 'imputation_metric':
                    metric.reset_trainer(self.trainer)
                    res_dic = metric.compute()
                    new_dic = {metric_tag + '_' + k: v for k, v in res_dic.items()}
                    outputs.append(new_dic)

        # Here outputs is a list of dictionnaries
        # We want to make sure that by merging them into a new dict
        # We do not loose keys
        res = {k: v for d in outputs for k, v in d.items()}
        assert len(res) == sum([len(di) for di in outputs])
        return res

    def multi_train(self, n_experiments, n_epochs, corruption=None, **kwargs):
        all_res = []
        for exp in range(n_experiments):
            all_res.append(self.train(n_epochs, corruption, **kwargs))
        self.res_data = pd.DataFrame(all_res)

    def write_csv(self, save_path):
        if self.res_data is None:
            raise AttributeError('No experiments yet. Use multitrain')
        self.res_data.to_csv(save_path, sep='\t')


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
    USE_BATCHES = False
    MY_DATASET = CortexDataset()
    N_EXPERIMENTS = 20
    N_EPOCHS = 120
    N_LL_MC_SAMPLES = 25
    MY_METRICS = [
        ('ll', LikelihoodMetric(trainer=None, n_mc_samples=N_LL_MC_SAMPLES)),
        ('imputation', ImputationMetric(trainer=None)),
        ('t_dropout', SummaryStatsMetric(trainer=None, stat_name='tstat', phi_name='dropout')),
        ('t_cv', SummaryStatsMetric(trainer=None, stat_name='tstat', phi_name='cv')),
    ]


    def my_model_fn(reconstruction_loss='zinb'):
        return VAE(MY_DATASET.nb_genes, n_batch=MY_DATASET.n_batches * USE_BATCHES,
                   dropout_rate=0.2, reconstruction_loss=reconstruction_loss)

    def zinb_model():
        return my_model_fn('zinb')

    def nb_model():
        return my_model_fn('nb')

    zinb_eval = ModelEval(model_fn=zinb_model,
                          dataset=MY_DATASET,
                          metrics=MY_METRICS)
    zinb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform')

    nb_eval = ModelEval(model_fn=nb_model,
                        dataset=MY_DATASET,
                        metrics=MY_METRICS)
    nb_eval.multi_train(n_experiments=N_EXPERIMENTS, n_epochs=N_EPOCHS, corruption='uniform')
