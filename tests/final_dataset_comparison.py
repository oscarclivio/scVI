#!/usr/bin/env python
# coding: utf-8


import glob
import pandas as pd
import numpy as np
import os
from statsmodels.stats.weightstats import ttest_ind



data_path = '/home/oscar/scVI/ppc/scripts/'

dataset_names = ['corr_zinb_dataset', 'corr_zifa_dataset', 'corr_nb_dataset']

for dataset_name in dataset_names:
    data_files = sorted(glob.glob(os.path.join(data_path, '{}/*.csv'.format(dataset_name))))
    data_names = ['nb', 'zifa_full', 'zinb']


    dfs = []
    for data_name, f in zip(data_names, data_files):
        my_df = pd.read_csv(f, sep='\t')
        my_df.loc[:, 'model'] = data_name.replace("_full","")
        dfs.append(my_df)
    df = pd.concat(dfs, axis=0)

    data_names[1] = 'zifa'


    metrics = ['ll_ll', 'imputation_median_imputation_score', 't_dropout_ks_stat', 't_ratio_ks_stat', 't_cv_ks_stat']
    h1_hypothesis = ['larger', 'larger', 'larger', 'larger', 'larger']
    h1_hypothesis_bis = ['smaller', 'smaller', 'smaller', 'smaller', 'smaller']

    df_nb = df.loc[df.model=='nb', metrics]
    df_zinb = df.loc[df.model=='zinb', metrics]




    def get_pvals(gby, other_df):
        my_df = gby[metrics]
        assert my_df.shape[1] == len(metrics)
        pvals = []
        for idx, alternative in enumerate(h1_hypothesis):
            assert len(other_df.values[:, idx]) != len(h1_hypothesis), (len(other_df.values[:, idx]), len(h1_hypothesis))
            _, pval, _ = ttest_ind(other_df.values[:, idx], my_df.values[:, idx], alternative=alternative)
            pvals.append(pval)
        return np.array(pvals)

    pvals_against_zinb = df.groupby('model').apply(get_pvals, other_df=df_zinb)
    pvals_against_zinb = (pvals_against_zinb
             .apply(lambda x: pd.Series(x))
             .T)
    pvals_against_zinb = pvals_against_zinb.rename(index={idx: met for (idx,met) in enumerate(metrics)})

    pvals_against_nb = df.groupby('model').apply(get_pvals, other_df=df_nb)
    pvals_against_nb = (pvals_against_nb
             .apply(lambda x: pd.Series(x))
             .T)
    pvals_against_nb = pvals_against_nb.rename(index={idx: met for (idx,met) in enumerate(metrics)})



    def get_summary(gby):
        res = {}
        res['mean'] = gby.mean()
        res['std'] = gby.std()
        res['pvals_against_nb'] = pvals_against_nb[gby.name]
        res['pvals_against_zinb'] = pvals_against_zinb[gby.name]
        return pd.DataFrame(res).T

    df_summary = df.groupby('model')[metrics].apply(get_summary)

    def my_styler(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
    #     print(val)
        style = ['background-color: yellow' if v < 0.05 else '' for v in val]
        return style


    print("\n======\nDataset : ", dataset_name, "\n")
    df_summary = df_summary.stack().unstack(1).sort_index(level=1).swaplevel()
    for model_against in ['nb','zinb']:
        for model in data_names:
            if model != model_against:

                name = '{}_beats_{}_strong'.format(model, model_against)
                df_summary.loc[:,name] = (df_summary['pvals_against_'+model_against] < 0.05)\
                                         & (df_summary.index.get_level_values('model') == model)
                print(name, ":", df_summary.loc[:,name].sum(), " -> ",
                      [a for a,_ in df_summary.index[df_summary.loc[:,name] == True].tolist()])

    print(" ")
    for model_against in data_names:
        for model in data_names:
            if model != model_against:

                name_weak = '{}_beats_{}_weak'.format(model, model_against)
                beats_weak_list = []

                for metric in metrics:

                    mean_model = df.loc[df.model == model, metric].mean()
                    mean_model_against = df.loc[df.model == model_against, metric].mean()

                    if mean_model < mean_model_against:
                        beats_weak_list.append(metric)

                print(name_weak, ":", len(beats_weak_list), " -> ", beats_weak_list)



    df_summary.to_csv(os.path.join(data_path, 'summary/{}.csv'.format(dataset_name)))






