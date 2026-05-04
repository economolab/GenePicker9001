# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 12:49:22 2025

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

# %%

# meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
# exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
# freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

meta = pd.read_csv(os.path.join(params.local_data_dir, "MO-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "MO-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "MO-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=3)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=10000)

# %% which genes are most correlated with a given gene

target_gene = "Slc6a5"

all_genes = ABC_utils.load_gene("scRNAseq")
gene_idx = np.where(all_genes == target_gene)[0][0]

gene_corrs = {}
for i in tqdm(range(len(all_genes))):
    check_gene = all_genes[i]
    
    x = exp_boot[:,[gene_idx, i]]
    R = np.corrcoef(x, rowvar=False)
    
    gene_corrs[check_gene] = R[0,1]

gene_corrs_df = {}
gene_corrs_df['gene'] = list(gene_corrs.keys())
gene_corrs_df['corrcoef'] = list(gene_corrs.values())
gene_corrs_df = pd.DataFrame(data=gene_corrs_df)

# %%

ABC_plot.plot_gene("Slc6a5", meta_boot, exp_boot, level='cluster')

# %%

ABC_plot.plot_gene("Slc30a3", meta_boot, exp_boot, level='cluster')

# %%

np.save("slc6a5_corr_genes.npy", gene_corrs_df['gene'][gene_corrs_df["corrcoef"] > 0].values)
    

# %%

import matplotlib.pyplot as plt 

gene_df = pd.read_csv("final_gene_df_100_MO_first_pass.csv")

genes = gene_df['gene'].values
all_genes = ABC_utils.load_gene("scRNAseq")

gene_idx = []
for gene in genes:
    gene_idx.append(np.where(all_genes == gene)[0][0])

import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, idx_genes, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to each other 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    idx_genes = idx_genes[idx]
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_genes


R = np.corrcoef(exp_boot[:,gene_idx].T)
R, idx_genes = cluster_corr(R, genes)


fig, ax = plt.subplots()
mappable = ax.imshow(R)

plt.gca().set_xticks(range(len(idx_genes)), idx_genes, rotation=45)
plt.gca().set_yticks(range(len(idx_genes)), idx_genes, rotation=45)
mappable.set_clim(-1, 1)
cbar = plt.colorbar(mappable)
cbar.set_label("PCC", rotation=270, labelpad=15)


# %%

ABC_plot.plot_gene("Ntng1", meta_boot, exp_boot, level='cluster')

# %%

gene_dfs = []
gene_df_names = ["final_gene_df_100_noset_20gene.csv",
                 "final_gene_df_100_set4_20gene.csv",
                 "final_gene_df_100_set6_20gene.csv",
                 "final_gene_df_100_set8_20gene.csv",
                 "final_gene_df_100_set9_20gene.csv",
                 "final_gene_df_100_set10_20gene.csv"]

for gene_df_name in gene_df_names:
    gene_df = pd.read_csv(gene_df_name)
    gene_df.sort_values(by='score', ascending=False, inplace=True)
    gene_dfs.append(gene_df)
    
tested_genes = []
for gene_df in gene_dfs:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)

places = np.zeros((len(tested_genes), len(gene_dfs)))

# iterate through genes
for i in range(np.shape(places)[0]):
    
    gene = tested_genes[i]
    
    # iterate through runs
    for j in range(np.shape(places)[1]):
        gene_df = gene_dfs[j]
        if gene in gene_df['gene'].values:
            places[i,j] = np.where(gene_df["gene"] == gene)[0][0]
        else:
            places[i,j] = len(tested_genes)
            
avg_places = np.mean(places, axis=1)
places_df = {}
places_df["gene"] = tested_genes
places_df["avg_place"] = avg_places
places_df = pd.DataFrame(data=places_df) 
           
fig, ax = plt.subplots()
for i in range(np.shape(places)[0]):
    ax.plot(range(len(gene_dfs)), places[i,:])

# %%

import matplotlib.pyplot as plt 

genes = tested_genes
all_genes = ABC_utils.load_gene("scRNAseq")

gene_idx = []
for gene in genes:
    gene_idx.append(np.where(all_genes == gene)[0][0])

import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, idx_genes, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to each other 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    idx_genes = idx_genes[idx]
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_genes


R = np.corrcoef(exp_boot[:,gene_idx].T)
R, idx_genes = cluster_corr(R, genes)


fig, ax = plt.subplots()
mappable = ax.imshow(R)

plt.gca().set_xticks(range(len(idx_genes)), idx_genes, rotation=45)
plt.gca().set_yticks(range(len(idx_genes)), idx_genes, rotation=45)
mappable.set_clim(-1, 1)
cbar = plt.colorbar(mappable)
cbar.set_label("PCC", rotation=270, labelpad=15)
        


