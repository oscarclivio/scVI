import numpy as np
from scvi.dataset import GeneExpressionDataset


class SyntheticDataset(GeneExpressionDataset):
    def __init__(self, batch_size=200, nb_genes=100, n_batches=2, n_labels=3, mode='zi',
                 nb_genes_zi=None):
        np.random.seed(0)
        # Generating samples according to a ZINB process
        data = np.random.negative_binomial(5, 0.3, size=(n_batches, batch_size, nb_genes))

        if mode == "zi":
            mask = np.random.binomial(n=1, p=0.7, size=(n_batches, batch_size, nb_genes))
            newdata = (data * mask)  # We put the batch index first
        elif mode == 'nb':
            newdata = data
        else:
            assert nb_genes_zi is not None
            assert nb_genes_zi <= nb_genes
            submask = np.random.binomial(n=1, p=0.7, size=(n_batches, batch_size, nb_genes_zi))
            mask = np.ones_like(data)
            mask[:, :, :nb_genes_zi] = submask
            newdata = data*mask

        labels = np.random.randint(0, n_labels, size=(n_batches, batch_size, 1))
        super().__init__(
            *GeneExpressionDataset.get_attributes_from_list(newdata, list_labels=labels),
            gene_names=np.arange(nb_genes).astype(np.str))


class NBDataset(SyntheticDataset):
    def __init__(self):
        super().__init__(mode='nb')


class ZINBDataset(SyntheticDataset):
    def __init__(self):
        super().__init__(mode='zi')


class Mixed25Dataset(SyntheticDataset):
    def __init__(self):
        super().__init__(mode='mixed', nb_genes=100, nb_genes_zi=25)


class Mixed50Dataset(SyntheticDataset):
    def __init__(self):
        super().__init__(mode='mixed', nb_genes=100, nb_genes_zi=50)


class Mixed75Dataset(SyntheticDataset):
    def __init__(self):
        super().__init__(mode='mixed', nb_genes=100, nb_genes_zi=75)
