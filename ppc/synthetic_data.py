import numpy as np
from scvi.dataset import GeneExpressionDataset


class SyntheticDatasetCorrelated(GeneExpressionDataset):
    def __init__(self, n_cells_cluster=100, n_clusters=5,
                 n_genes_high=15, n_genes_total=50,
                 weight_high=0.1, weight_low=0.01,
                 lam_pure=100., p_dropout=0.2,
                 n_batches=1, n_labels=1, mode='mixed', ratio_genes_zi=0.5):
        """

        :param n_cells_cluster:
        :param n_clusters:
        :param n_genes_high:
        :param n_genes_total:
        :param weight_high:
        :param weight_low:
        :param lam_pure:
        :param p_dropout:
        :param n_batches:
        :param n_labels:
        :param mode:
        :param ratio_genes_zi:
        """

        assert mode in ['mixed', 'zi', 'nb'], 'Mode {} not recognized'.format(mode)
        np.random.seed(0)
        
        if n_genes_total % n_clusters > 0:
            print("Warning, clusters have inequal sizes")
            
        if n_genes_high > (n_genes_total // n_clusters):
            print("Overlap of", n_genes_high - (n_genes_total // n_clusters), "genes")

        # Generate data before dropout 
        batch_size = n_cells_cluster * n_clusters
        data = np.ones((n_batches, batch_size, n_genes_total))
        
        # "Pure" data, before noise
        # For each cell cluster, some genes have a high expression, the rest
        # has a low expression. The scope of high expression genes "moves"
        # with the cluster
        for cluster in range(n_clusters):
            ind_first_gene_cluster = cluster * (n_genes_total // n_clusters)
            ind_last_high_gene_cluster = ind_first_gene_cluster + n_genes_high
            
            # Weights in a cluster to create highly-expressed and low-expressed genes
            weights = weight_low * np.ones((n_genes_total,))
            weights[ind_first_gene_cluster:ind_last_high_gene_cluster] = weight_high
            weights /= weights.sum()
            exprs = np.random.poisson(lam_pure*weights, size=(n_batches, n_cells_cluster,
                                                              n_genes_total))
            data[:, cluster*n_cells_cluster:(cluster+1)*n_cells_cluster, :] = exprs
        # Apply dropout depending on the mode
        if mode == "zi":
            mask = np.random.binomial(n=1, p=1-p_dropout, size=(n_batches, batch_size, n_genes_total))
            newdata = (data * mask)  # We put the batch index first
        elif mode == 'nb':
            newdata = data
        else:
            assert ratio_genes_zi is not None
            assert ratio_genes_zi <= 1.
            n_genes_zi = int(n_genes_total * ratio_genes_zi)

            submask = np.random.binomial(n=1, p=1-p_dropout, size=(n_batches, batch_size, n_genes_zi))
            mask = np.ones_like(data)
            
            random_permutation = np.random.permutation(np.arange(n_genes_total))
            self.zi_genes_idx = random_permutation[:n_genes_zi]
            mask[:, :, random_permutation[:n_genes_zi]] = submask

            newdata = data*mask
        labels = np.random.randint(0, n_labels, size=(n_batches, batch_size, 1))
        super().__init__(
            *GeneExpressionDataset.get_attributes_from_list(newdata, list_labels=labels),
            gene_names=np.arange(n_genes_total).astype(np.str))


class CorrNBDataset(SyntheticDatasetCorrelated):
    def __init__(self):
        super().__init__(mode='nb')


class CorrZINBDataset(SyntheticDatasetCorrelated):
    def __init__(self):
        super().__init__(mode='zi')


class CorrMixed25Dataset(SyntheticDatasetCorrelated):
    def __init__(self):
        super().__init__(mode='mixed', ratio_genes_zi=0.25)


class CorrMixed50Dataset(SyntheticDatasetCorrelated):
    def __init__(self):
        super().__init__(mode='mixed', ratio_genes_zi=0.50)


class CorrMixed75Dataset(SyntheticDatasetCorrelated):
    def __init__(self):
        super().__init__(mode='mixed', ratio_genes_zi=0.75)



###########################################
class SyntheticDataset(GeneExpressionDataset):
    def __init__(self, batch_size=4000, nb_genes=1200, n_batches=1, n_labels=3, mode='zi',
                 ratio_genes_zi=None):
        np.random.seed(0)
        # Generating samples according to a ZINB process
        data = np.random.negative_binomial(5, 0.3, size=(n_batches, batch_size, nb_genes))

        if mode == "zi":
            mask = np.random.binomial(n=1, p=0.7, size=(n_batches, batch_size, nb_genes))
            newdata = (data * mask)  # We put the batch index first
        elif mode == 'nb':
            newdata = data
        else:
            assert ratio_genes_zi is not None
            assert ratio_genes_zi <= 1.
            nb_genes_zi = int(nb_genes * ratio_genes_zi)
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
        super().__init__(mode='mixed', ratio_genes_zi=.25)


class Mixed50Dataset(SyntheticDataset):
    def __init__(self):
        super().__init__(mode='mixed', ratio_genes_zi=.5)


class Mixed75Dataset(SyntheticDataset):
    def __init__(self):
        super().__init__(mode='mixed', ratio_genes_zi=.75)


