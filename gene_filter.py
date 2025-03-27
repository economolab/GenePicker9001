# -*- coding: utf-8 -*-
"""
Filter gene pool using heuristics
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from ABC_toolbox import ABC_utils, gene_funcs

def filt_genes(exp, meta, freqs, top_n=200, rdr_weight=(1/3), var_E_weight=(2/3)):
    
    genes = ABC_utils.load_gene("scRNAseq")
    
    rdrs = gene_funcs.calc_rdr(exp, meta, freqs)
    
    # set Malat1 RDR to the second highest RDR because it's always a bit of an outlier
    Malat1_idx = np.where(genes == 'Malat1')[0][0]
    second_highest = np.max(rdrs[np.arange(len(rdrs)) != Malat1_idx])
    rdrs[Malat1_idx] = second_highest
    
    var_Es = gene_funcs.calc_var_E(exp, meta, freqs, N=2)
    
    data_dict = {}

    data_dict['gene'] = genes
    data_dict['rdr'] = rdrs
    data_dict['var_E'] = var_Es

    gene_df = pd.DataFrame(data=data_dict)

    rdr_scaler = MinMaxScaler()
    var_E_scaler = MinMaxScaler()

    rdr_scaler.fit(gene_df["rdr"].values.reshape(-1, 1))
    var_E_scaler.fit(gene_df["var_E"].values.reshape(-1, 1))

    gene_df["rdr"] = rdr_scaler.transform(gene_df["rdr"].values.reshape(-1, 1))
    gene_df["var_E"] = var_E_scaler.transform(gene_df["var_E"].values.reshape(-1, 1))
    
    weights = [rdr_weight, var_E_weight]

    arr = np.column_stack((gene_df["rdr"], gene_df["var_E"]))
    scores = np.average(arr, axis=1, weights=weights)

    ind = np.argpartition(scores, -top_n)[-top_n:]

    gene_df["top_gene"] = np.isin(range(len(gene_df)), ind)
    top_genes = gene_df["gene"][ind].values
    
    return gene_df, top_genes



    
