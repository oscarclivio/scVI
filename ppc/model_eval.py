from scvi.inference.inference import UnsupervisedTrainer
from scvi.dataset import CortexDataset, RetinaDataset, HematoDataset, PbmcDataset, \
    BrainSmallDataset
from scvi.models import VAE

import copy
import pandas as pd
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


if __name__ == '__main__':
    use_batches = False
    my_dataset = CortexDataset()

    my_metrics = [
        ('ll', LikelihoodMetric(trainer=None, n_mc_samples=10)),
        ('imputation', ImputationMetric(trainer=None)),
        ('t_dropout', SummaryStatsMetric(trainer=None, stat_name='tstat', phi_name='dropout')),
        ('t_cv', SummaryStatsMetric(trainer=None, stat_name='tstat', phi_name='cv')),
    ]


    def my_model_fn(reconstruction_loss='zinb'):
        return VAE(my_dataset.nb_genes, n_batch=my_dataset.n_batches * use_batches,
                   dropout_rate=0.2, reconstruction_loss=reconstruction_loss)


    def zimb_model():
        return my_model_fn('zinb')


    zinb_eval = ModelEval(model_fn=zimb_model,
                          dataset=my_dataset,
                          metrics=my_metrics)
    zinb_eval.multi_train(n_experiments=5, n_epochs=5, corruption='uniform')
