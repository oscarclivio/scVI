from functools import partial
from scipy.stats import ttest_1samp, ks_2samp
import numpy as np
import torch


class Metric:
    def __init__(self, trainer, tag="", phi_name=None, n_sample_posterior=25, batch_size=32):
        """

        :param trainer: scVI Trainer object instance
        :param phi_name: Name of the phi function used (optionnal, depending on Metric)
        :param n_sample_posterior: Number of times the posterior is sampled
        :param batch_size: Batch size for Posterior sampling
        """

        phi_tag = '_' + phi_name if phi_name is not None else ''
        self.name = "abstract_metric{}".format(phi_tag)
        self.trainer = trainer
        self.phi = None
        self.n_sample_posterior = n_sample_posterior
        self.batch_size = batch_size
        self.init_phi(phi_name=phi_name)
        
        self.tag = tag
        self.keys = []

    def compute(self):
        pass

    def _csv(self):
        pass

    def plot(self):
        pass

    @torch.no_grad()
    def generate(self):
        is_zero_inflated = self.trainer.model.reconstruction_loss == 'zinb'
        x_gen, x_real = self.trainer.train_set.generate(genes=None,
                                                        n_samples=self.n_sample_posterior,
                                                        zero_inflated=is_zero_inflated,
                                                        batch_size=self.batch_size)
        return x_gen.squeeze(), x_real

    def output_dict(self, dict_to_output):
        
        new_dict = {self.tag + "_" + key: value for key,value in dict_to_output.items()}
        
        self.keys = list(new_dict.keys())
            
        return new_dict
    
        

    @staticmethod
    def phi_ratio(array, axis=None):
        """
        Computes ratio of zeros / average of non zeros
        :param array:
        :param axis:
        :return:
        """
        nb_zeros = (array.astype(int) == 0).sum(axis=axis)
        nb_non_zeros = (array.astype(int) != 0).sum(axis=axis)
        avg_expression = 1e-4 * np.ones(nb_non_zeros.shape)
        if (nb_non_zeros == 0).sum() > 0:
            print("hum... Zeros rows are here !! Size :", (nb_non_zeros == 0.).sum())
        avg_expression[nb_non_zeros != 0] = array.sum(axis=axis)[nb_non_zeros != 0] / nb_non_zeros[nb_non_zeros != 0]
        return nb_zeros / avg_expression

    @staticmethod
    def phi_dropout(array, axis=None, do_mean=True):
        """
        Computes Number / Mean of zeros
        :param array:
        :param axis:
        :param do_mean:
        :return:
        """
        if do_mean:
            zeros_metric = (array.astype(int) == 0).mean(axis=axis)
        else:
            zeros_metric = (array.astype(int) == 0).sum(axis=axis)
        return zeros_metric

    @staticmethod
    def phi_cv(array, axis=None):
        """
        Computes coef of Variation
        :param array:
        :param axis:
        :return:
        """
        mean = array.mean(axis=axis)
        if (mean == 0.).sum() > 0:
            print("Zeros rows are here !! Size :", (mean == 0.).sum())
        mean[mean == 0.] = 1e-4
        return array.std(axis=axis) / mean

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
        """
        Useful because you may want to average the metric on several trainings
        :param new_trainer:
        :return:
        """
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
        return self.output_dict({'ll': ll})


class ImputationMetric(Metric):
    def __init__(self, n_samples_imputation=1, **kwargs):
        super().__init__(**kwargs)
        self.name = "imputation_metric"
        self.n_samples_imputation = n_samples_imputation

    def compute(self):
        original_list, imputed_list = self.trainer.train_set.imputation_benchmark(verbose=False,
                                                                                  n_samples=self.n_samples_imputation,
                                                                                  show_plot=False)
        imputation_errors = np.abs(np.concatenate(original_list) - np.concatenate(imputed_list))
        median_imputation_score = np.median(imputation_errors)

        return self.output_dict({'median_imputation_score': median_imputation_score})


class DifferentialExpressionMetric(Metric):
    def __init__(self, n_samples=300, M_permutation=40000, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.M_permutation = M_permutation
        self.name = "differential_expression_metric"

    def compute(self):
        results = {}
        results['bayes_factor_avg_per_gene_same_cells'] = []
        results['bayes_factor_avg_per_gene_diff_cells'] = []
        results['ratio_genes_detected_per_gene_diff_cells'] = []
        all_labels = np.unique(self.trainer.gene_dataset.labels)
        all_labels_significant = np.array(
            [val for val in all_labels if (self.trainer.gene_dataset.labels == val).sum() >= 2])
        for label1 in all_labels_significant:
            for label2 in all_labels_significant:
                cell_a_idx = (self.trainer.gene_dataset.labels == label1)
                cell_b_idx = (self.trainer.gene_dataset.labels == label2)
                st = self.trainer.train_set.differential_expression_score(cell_a_idx, cell_b_idx,
                                                                          n_samples=self.n_samples,
                                                                          M_permutation=self.M_permutation,
                                                                          all_stats=False)

                if label1 == label2:
                    results['bayes_factor_avg_per_gene_same_cells'].append(np.abs(st))

                else:
                    results['bayes_factor_avg_per_gene_diff_cells'].append(np.abs(st))
                    results['ratio_genes_detected_per_gene_diff_cells'].append((np.abs(st) >= 3))

        means_results = {'avg_' + key_res: np.mean(np.array(list_res), axis=0)
                         for key_res, list_res in results.items()}

        return self.output_dict(means_results)


class SummaryStatsMetric(Metric):
    def __init__(self, stat_name='tstat', **kwargs):
        """

        :param stat_name:
        if name is 'tstatr' ==> one statistic, pval for EACH GENE
        if name is 'ks' ==> Corresponds to Posterior Predictive Check using Kolmogorov
        Smirnov Test


        :param kwargs:
        """
        super().__init__(**kwargs)
        assert self.phi is not None
        self.name = "summary_stats_metric"
        self.stat_name = stat_name

    def compute(self):
        x_gen, x_real = self.generate()
        phi_real_gene = self.phi(x_real, axis=0)  # n_genes \times n_sim
        phi_gen_gene = self.phi(x_gen, axis=0)

        if self.stat_name == 'tstat':
            # Computed for EACH gene
            stat_phi, pvals = ttest_1samp(phi_real_gene, phi_gen_gene, axis=0)
            return self.output_dict({"tstat_phi": stat_phi,
                                     "t_pvals": pvals,
                                     "phi_gen_gene": phi_gen_gene,
                                     "phi_real_gene": phi_real_gene})

        elif self.stat_name == 'ks':
            # Computed accross ALL genes
            phi_gen_gene_avg = phi_gen_gene.mean(axis=-1)
            assert phi_gen_gene_avg.shape == phi_real_gene.shape, \
                (phi_gen_gene_avg.shape, phi_real_gene.shape)
            ks_stat, pval = ks_2samp(phi_gen_gene_avg, phi_real_gene)
            return self.output_dict({
                "ks_stat": ks_stat,
                "ks_pval": pval,
                "phi_gen_gene": phi_gen_gene,
                "phi_real_gene": phi_real_gene})
        else:
            raise AttributeError('{} is not a valid statistic choice.', self.stat_name)


class OutlierStatsMetric(SummaryStatsMetric):
    """
    USELESS EASIER TO START FROM SUMMARYSTATSMETRIC
    """
