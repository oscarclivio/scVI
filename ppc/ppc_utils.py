from scvi.dataset import CortexDataset, RetinaDataset, HematoDataset, PbmcDataset, \
    BrainSmallDataset
from scipy.stats.mstats import ks_2samp
import numpy as np

datasets_mapper = {
    'pbmc': PbmcDataset,
    'cortex': CortexDataset,
    'retina': RetinaDataset,
    'hemato': HematoDataset,
    'brain_small': BrainSmallDataset
}


def phi_ratio(array, axis=None):
    nb_zeros = (array.astype(int) == 0).sum(axis=axis)
    nb_non_zeros = (array.astype(int) != 0).sum(axis=axis)
    avg_expression = array.sum(axis=axis) / nb_non_zeros
    return nb_zeros / avg_expression


def phi_dropout(array, axis=None, do_mean=True):
    if do_mean:
        zeros_metric = (array.astype(int) == 0).mean(axis=axis)
    else:
        zeros_metric = (array.astype(int) == 0).sum(axis=axis)
    return zeros_metric


def phi_dropout_sum(array, axis=None):
    return phi_dropout(array, axis, do_mean=False)


def phi_cv(array, axis=None):
    return array.std(axis=axis) / array.mean(axis=axis)



phi_mapper = {
    'ratio': phi_ratio,
    'dropout': phi_dropout,
    'dropout_sum': phi_dropout_sum,
    'cv': phi_cv,
}


def compute_ks_ppc(real_data, gen_data):
    ks = []
    for col in range(gen_data.shape[1]):
        ks_iter, _ = ks_2samp(real_data, gen_data[:, col])
        ks.append(ks_iter)
    return np.array(ks)
