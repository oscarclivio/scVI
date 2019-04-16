from functools import partial
from scipy.stats import ttest_1samp
import pandas as pd


class Metric:
    def __init__(self, trainer, phi_name=None):
        self.trainer = trainer
        self.phi = None

        self.init_phi(phi_name=phi_name)

    def compute(self):
        pass

    def save_csv(self):
        pass

    def plot(self):
        pass

    def generate(self, n_sample_posterior=25, batch_size=32):
        x_gen, x_real = self.trainer.train_set.generate(genes=None,
                                                        n_samples=n_sample_posterior,
                                                        zero_inflated=False,
                                                        batch_size=batch_size)
        return x_gen, x_real

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


class LikelihoodMetric(Metric):
    def __init__(self, trainer):
        super().__init__(trainer=trainer)


class ImputationMetric(Metric):
    def __init__(self, trainer):
        super().__init__(trainer=trainer)


class DifferentialExpressionMetric(Metric):
    def __init__(self, trainer):
        super().__init__(trainer=trainer)


class SummaryStatsMetric(Metric):
    def __init__(self, trainer, phi_name, stat_name='tstat'):
        super().__init__(trainer=trainer, phi_name=phi_name)
        self.stat = None

        self.init_stat(stat_name)

    def compute(self, n_sample_posterior=25, batch_size=32):
        x_gen, x_real = self.generate(n_sample_posterior, batch_size)
        phi_real_gene = self.phi(x_real, axis=0)
        phi_gen_gene = self.phi(x_gen, axis=0)

        stat_phi, pvals = self.stat(phi_real_gene, phi_gen_gene, axis=0)
        return stat_phi, pvals

    def init_stat(self, stat_name):
        if stat_name == 'tstat':
            self.stat = ttest_1samp


class OutlierStatsMetric(Metric):
    def __init__(self, trainer):
        super().__init__(trainer=trainer)

