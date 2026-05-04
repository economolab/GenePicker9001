# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 18:15:30 2025

@author: jpv88
"""

import params
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ABC_toolbox import ABC_utils, cell_funcs

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-norm.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=3)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=5000)

# %%

genes = ['Phox2b', 'Slc17a6', 'Slc6a5', 'Slc5a7', 'Zfhx3', 'Meis2', 'Tenm2', 
         'Tshz2', 'Hoxb5', 'Ebf3', 'Ralyl', 'Rph3a', 'Syt1', 'Alcam', 'Arpp21', 'Pcp4']

for gene in genes:
    
    gene_mask = (ABC_utils.load_gene("scRNAseq") == gene)
    exp_gene = exp_boot[:,gene_mask]
    fig, ax = plt.subplots()
    plt.hist(exp_gene, bins=30)
    plt.title(gene)
    
# %%

from more_itertools import distinct_combinations, set_partitions
from tqdm import tqdm

genes = ['Phox2b', 'Zfhx3', 'Meis2', 'Tenm2', 'Tshz2', 'Ralyl', 'Alcam', 'Arpp21', 'Pcp4']

genes_d = {}
for gene in genes:
    gene_mask = (ABC_utils.load_gene("scRNAseq") == gene)
    genes_d[gene] = np.squeeze(exp_boot[:,gene_mask])
    
genes_df = pd.DataFrame(data=genes_d)
corr_df = genes_df.corr()

# return average correlation between genes in a group of genes
def corr_group(corr_df, group):
    
    combinations = list(distinct_combinations(group, 2))
    
    corrs = []
    for combination in combinations:
        corrs.append(corr_df[combination[0]].loc[combination[1]])
    
    # take the two lowest values
    corrs.sort()
    corrs = corrs[:2]
        
    return np.max(corrs)

# calculate average correlation of a partition
def corr_part(corr_df, part):
    
    corrs = []
    for group in part:
        corrs.append(corr_group(corr_df, group))
    
    return np.max(corrs)
    
parts = []
for part in set_partitions(genes, k=3, min_size=3, max_size=3):
    parts.append(part)
    
corrs_d = {}
for part in tqdm(parts):
    corrs_d[str(part)] = corr_part(corr_df, part)
    




