from . import GeneExpressionDataset
from . import AnnDataset
import torch
import pickle
import os
import numpy as np

import pandas as pd
import anndata


class AnnDatasetKeywords(GeneExpressionDataset):

    def __init__(self, data, select_genes_keywords=[]):

        if isinstance(data, str):
            anndataset = anndata.read(data)
        else:
            anndataset = data

        idx_and_gene_names = [(idx, gene_name) for idx, gene_name in enumerate(list(anndataset.var.index))]
        for keyword in select_genes_keywords:
            idx_and_gene_names = [(idx, gene_name) for idx, gene_name in idx_and_gene_names
                                  if keyword.lower() in gene_name.lower()]

        gene_indices = np.array([idx for idx, _ in idx_and_gene_names])
        gene_names = np.array([gene_name for _, gene_name in idx_and_gene_names])

        expression_mat = np.array(anndataset.X[:, gene_indices].todense())

        select_cells = expression_mat.sum(axis=1) > 0
        expression_mat = expression_mat[select_cells, :]

        select_genes = (expression_mat > 0).mean(axis=0) > 0.21
        gene_names = gene_names[select_genes]
        expression_mat = expression_mat[:, select_genes]

        print("Final dataset shape :", expression_mat.shape)

        super(AnnDatasetKeywords, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(expression_mat),
            gene_names=gene_names
        )




class ZhengDataset(AnnDatasetKeywords):

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        zheng = anndata.read(os.path.join(current_dir, 'zheng_gemcode_control.h5ad'))

        super(ZhengDataset, self).__init__(zheng, select_genes_keywords=['ercc'])


class MacosDataset(AnnDatasetKeywords):

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        macos = anndata.read(os.path.join(current_dir, 'macosko_dropseq_control.h5ad'))

        super(MacosDataset, self).__init__(macos, select_genes_keywords=['ercc'])

class KleinDataset(AnnDatasetKeywords):

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        klein = anndata.read(os.path.join(current_dir, 'klein_indrops_control_GSM1599501.h5ad'))

        super(KleinDataset, self).__init__(klein, select_genes_keywords=['ercc'])

class Sven1Dataset(AnnDatasetKeywords):

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(os.path.join(current_dir, 'svensson_chromium_control.h5ad'))

        sven1 = svens[svens.obs.query('sample == "20311"').index]
        super(Sven1Dataset, self).__init__(sven1, select_genes_keywords=['ercc'])

class Sven2Dataset(AnnDatasetKeywords):

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(os.path.join(current_dir, 'svensson_chromium_control.h5ad'))

        sven2 = svens[svens.obs.query('sample == "20312"').index]
        super(Sven2Dataset, self).__init__(sven2, select_genes_keywords=['ercc'])
