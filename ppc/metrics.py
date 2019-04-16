from functools import partial
from scipy.stats import ttest_1samp
import numpy as np


class Metric:
    def __init__(self, trainer, phi_name=None, n_sample_posterior=25, batch_size=32):
        phi_tag = '_'+phi_name if phi_name is not None else ''
        self.name = "abstract_metric{}".format(phi_tag)
        self.trainer = trainer
        self.phi = None
        self.n_sample_posterior = n_sample_posterior
        self.batch_size = batch_size
        self.init_phi(phi_name=phi_name)

    def compute(self):
        pass

    def _csv(self):
        pass

    def plot(self):
        pass

    def generate(self):
        is_zero_inflated = self.trainer.model.reconstruction_loss == 'zinb'
        x_gen, x_real = self.trainer.train_set.generate(genes=None,
                                                        n_samples=self.n_sample_posterior,
                                                        zero_inflated=is_zero_inflated,
                                                        batch_size=self.batch_size)
        return x_gen.squeeze(), x_real

    @staticmethod
    def phi_ratio(array, axis=None):
        nb_zeros = (array.astype(int) == 0).sum(axis=axis)
        nb_non_zeros = (array.astype(int) != 0).sum(axis=axis)
        avg_expression = array.sum(axis=axis) / nb_non_zeros
        return nb_zeros / avg_expression

    @staticmethod
    def phi_dropout(array, axis=None, do_mean=True):
        if do_mean:
            zeros_metric = (array.astype(int) == 0).mean(axis=axis)
        else:
            zeros_metric = (array.astype(int) == 0).sum(axis=axis)
        return zeros_metric

    @staticmethod
    def phi_cv(array, axis=None):
        return array.std(axis=axis) / array.mean(axis=axis)

    def init_phi(self, phi_name):
        if phi_name == 'ratio':
            self.phi = self.phi_ratio
        elif phi_name == 'dropout':
            self.phi = self.phi_dropout
        elif phi_name == 'dropout_sum':
            self.phi = partial(self.phi_dropout, do_mean=False)
        elif phi_name == 'cv':
            self.phi = self.phi_cv

    def reset_trainer(self, new_trainer):
        self.trainer = new_trainer


class LikelihoodMetric(Metric):
    def __init__(self, verbose=False, n_mc_samples=1000, **kwargs):
        super().__init__(**kwargs)
        self.name = "likelihood"
        self.verbose = verbose
        self.n_mc_samples = n_mc_samples

    def compute(self):
        ll = self.trainer.test_set.marginal_ll(verbose=self.verbose,
                                               n_mc_samples=self.n_mc_samples)
        return {'ll': ll}


class ImputationMetric(Metric):
    def __init__(self, n_samples_imputation=1, **kwargs):
        super().__init__(**kwargs)
        self.name = "imputation_metric"
        self.n_samples_imputation = n_samples_imputation
        
    def compute(self):
        original_list, imputed_list = self.trainer.train_set.imputation_benchmark(verbose=True,
                                                                                  n_samples=self.n_samples_imputation,
                                                                                  show_plot=False)
        imputation_errors = np.abs(np.concatenate(original_list) - np.concatenate(imputed_list))
        median_imputation_score = np.median(imputation_errors)
        
        return {'median_imputation_score': median_imputation_score}


class DifferentialExpressionMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "differential_expression_metric"


class SummaryStatsMetric(Metric):
    def __init__(self, stat_name='tstat', **kwargs):
        super().__init__(**kwargs)
        assert self.phi is not None
        self.name = "summary_stats_metric"
        self.stat = None
        self.init_stat(stat_name)

    def compute(self):
        x_gen, x_real = self.generate()
        phi_real_gene = self.phi(x_real, axis=0)
        phi_gen_gene = self.phi(x_gen, axis=0)

        stat_phi, pvals = self.stat(phi_real_gene, phi_gen_gene, axis=0)
        return {"stat_phi": stat_phi, "pvals": pvals}

    def init_stat(self, stat_name):
        if stat_name == 'tstat':
            self.stat = ttest_1samp


class OutlierStatsMetric(SummaryStatsMetric):
    """
    USELESS EASIER TO START FROM SUMMARYSTATSMETRIC
    """
