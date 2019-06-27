from functools import partial
from scipy.stats import ttest_1samp, ks_2samp
import numpy as np
import torch
from scvi.models.log_likelihood import gene_specific_ll
import scipy


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

    @torch.no_grad()
    def compute(self):
        pass

    def _csv(self):
        pass

    def plot(self):
        pass

    @torch.no_grad()
    def generate(self):
        is_zero_inflated = self.trainer.model.reconstruction_loss == 'zinb'

        posterior = self.trainer.train_set
        x_gen, x_real = posterior.generate(genes=None,n_samples=self.n_sample_posterior,zero_inflated=is_zero_inflated,
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
            raise ValueError("Zeros rows are here !! Size : " + str( (nb_non_zeros == 0.).sum()))
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
            raise ValueError("Zeros rows are here !! Size : " + str( (mean == 0.).sum()))
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


class GeneSpecificLikelihoodMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "gene_specific_dropout"

    @torch.no_grad()
    def compute(self):
        lls = gene_specific_ll(self.trainer.model, self.trainer.test_set)

        return self.output_dict({'gene_ll': lls})


class GeneSpecificDropoutMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "gene_specific_likelihood"

    @torch.no_grad()
    def compute(self):
        outputs = {}
        if hasattr(self.trainer.model, 'reconstruction_loss'):
            if self.trainer.model.reconstruction_loss == 'zinb':

                train_set = self.trainer.train_set
                dropout_logits_train = []
                for tensors in train_set.update({"batch_size": 128}):
                    sample_batch, _, _, batch_index, labels = tensors
                    px_dropout = self.trainer.model.inference(sample_batch,batch_index=batch_index,
                                                      y=labels)[3]
                    dropout_logits_train.append(px_dropout.cpu().numpy())
                dropout_logits_train = np.concatenate(dropout_logits_train)
                dropout_probs_train = 1. / (1. + np.exp(-dropout_logits_train))

                test_set = self.trainer.test_set
                dropout_logits_test = []
                for tensors in test_set.update({"batch_size": 128}):
                    sample_batch, _, _, batch_index, labels = tensors
                    px_dropout = self.trainer.model.inference(sample_batch,batch_index=batch_index,
                                                      y=labels)[3]
                    dropout_logits_test.append(px_dropout.cpu().numpy())
                dropout_logits_test = np.concatenate(dropout_logits_test)
                dropout_probs_test = 1. / (1. + np.exp(-dropout_logits_test))

                dropout_logits_all = np.concatenate([dropout_logits_train, dropout_logits_test])
                dropout_probs_all = np.concatenate([dropout_probs_train, dropout_probs_test])

                outputs['gene_dropout_logits_train'] = dropout_logits_train.mean(axis=0)
                outputs['gene_dropout_logits_test'] = dropout_logits_test.mean(axis=0)
                outputs['gene_dropout_logits_all'] = dropout_logits_all.mean(axis=0)

                outputs['gene_dropout_probs_train'] = dropout_probs_train.mean(axis=0)
                outputs['gene_dropout_probs_test'] = dropout_probs_test.mean(axis=0)
                outputs['gene_dropout_probs_all'] = dropout_probs_all.mean(axis=0)


        return self.output_dict(outputs)


class InferParamsOnZerosMetric(Metric):

    def __init__(self, mask_zero=None, **kwargs):
        if isinstance(mask_zero,scipy.sparse.csr.csr_matrix):
            mask_zero = np.array(mask_zero.toarray())
        self.mask_zero = mask_zero
        super().__init__(**kwargs)
        self.name = "zero_infer_params"

    @torch.no_grad()
    def compute(self):
        outputs = {}
        if hasattr(self.trainer.model, 'reconstruction_loss'):
            if self.trainer.model.reconstruction_loss == 'zinb':

                posterior = self.trainer.create_posterior(self.trainer.model, self.trainer.gene_dataset,
                                                          indices=np.arange(len(self.trainer.gene_dataset)))

                dropout_logits_full = []
                scales_full = []
                rates_full = []
                for tensors in posterior:
                    sample_batch, _, _, batch_index, labels = tensors
                    px_scale, _, px_rate, px_dropout = self.trainer.model.inference(sample_batch,
                                                                                    batch_index=batch_index,
                                                                                    y=labels)[0:4]
                    scales_full.append(px_scale.cpu().numpy())
                    rates_full.append(px_rate.cpu().numpy())
                    dropout_logits_full.append(px_dropout.cpu().numpy())
                dropout_logits_full = np.concatenate(dropout_logits_full)
                scales_full = np.concatenate(scales_full)
                rates_full = np.concatenate(rates_full)
                dropout_probs_full = 1. / (1. + np.exp(-dropout_logits_full))

                outputs['zero_dropout_probs_all'] = dropout_probs_full[self.mask_zero]
                outputs['zero_scales_all'] = scales_full[self.mask_zero]
                outputs['zero_rates_all'] = rates_full[self.mask_zero]


        return self.output_dict(outputs)

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
