# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:43:52 2026

@author: jpv88
"""

import logging
import os
import pygad
import random

import numpy as np
import pandas as pd

import params

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from ABC_toolbox import ABC_plot, ABC_utils, cell_funcs, classify_cells, gene_funcs, cluster_recombination
from gene_filter import filt_genes
from genetic_algo import run_ga, find_genes

#%%

meta = pd.read_csv(os.path.join(params.local_data_dir, "MO-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "MO-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "MO-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

#%%

splits = cell_funcs.K_fold_cells(meta)
supers = cell_funcs.boot_super_splits(exp, meta, splits, k=3)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
boots = cell_funcs.bootstrap_scRNAseq_splits(supers, freqs, n=5000)

def merge_genes(exp_mat, merges):
    
    scRNAseq_genes = ABC_utils.load_gene("scRNAseq")
    gene_to_idx = {gene: i for i, gene in enumerate(scRNAseq_genes)}
    
    for a, b in merges:
        i, j = gene_to_idx[a], gene_to_idx[b]
        new_col = exp_mat[:,i] + exp_mat[:,j]
        exp_mat = np.column_stack((exp_mat, new_col))
    
    return exp_mat

merges = [['Cacna2d3', 'Pcp4'], ['Meis2', 'Thsd7a'], ['Cbln2', 'Calb1'], ['Pvalb', 'Lamp5']]

boots = [list(t) for t in boots]

for i in range(len(boots)):
    
    boots[i][0] = list(boots[i][0])
    boots[i][1] = list(boots[i][1])
    
    boots[i][0][0] = merge_genes(boots[i][0][0], merges)
    boots[i][1][0] = merge_genes(boots[i][1][0], merges)
    
data, freqs = classify_cells.preprocess_data_splits(boots, freqs)
        

#%%

genes_singleton = ['Stxbp6', 'Brinp3', 'Chrm3', 'Spon1', 'Galntl6', 'Meis2', 'Cbln2', 
         'Slc24a2', 'Cnr1', 'Grik1', 'Cpne4', 'Kcnc2', 'Pcp4', 'Alcam', 
         'Thsd7a', 'Calb1', 'Cdh13', 'Cck', 'Slc17a7', 'Slc32a1']

genes_merge = ['Slc17a7', 'Slc32a1', 'Stxbp6', 'Kcnk2', 'Brinp3', 'Chrm3', 'Spon1', 'Galntl6', 
               'Slc24a2', 'Cnr1', 'Grik1', 'Cpne4', 'Kcnc2', 'Alcam', 'Cdh13', 'Cck']

full_accs_singleton = []
for i in range(10):
    res = classify_cells.cross_val_classifier_splits(data, freqs, genes=genes_singleton, merges=merges)
    full_accs_singleton.extend(res['accs'])
    
full_accs_merge = []
for i in range(10):
    res = classify_cells.cross_val_classifier_splits(data, freqs, genes=genes_merge, merges=merges)
    full_accs_merge.extend(res['accs'])

#%%

def cluster_cm_to_all_accs():
    
    