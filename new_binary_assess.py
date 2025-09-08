# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:13:17 2025

@author: jpv88
"""

import os
import sys

# obtain this file's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

sys.path.append(dname)
import params

import numpy as np
import pandas as pd
import scipy as sp

import scanpy as sc
from anndata import AnnData, read_h5ad
from scipy import stats
from scipy.stats import iqr, variation
from tqdm import tqdm

from ABC_toolbox import cell_funcs, gene_funcs
from utils import data_nav, find_genes, classify_cells, utils

f_gene = os.path.join(params.data_dir, "metadata\\WMB-10X\\20231215")
f_gene = os.path.join(f_gene, "gene.csv")
all_gene = pd.read_csv(f_gene)
all_gene = all_gene["gene_symbol"].values

# %% load ant-IRN-PARN data

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

# %%

exp_super, meta_super = cell_funcs.boot_super(exp, meta)
# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=10000)

# %%

# cre_lines = ["Scnn1a", "Slc6a3", "Ntsr1", "Slc17a8", "Pvalb", "Slc6a4", "Crh", 
#              "Slc32a1", "Slc17a6", "Gal", "Dbh", "Cux2", "Rbp4", "Slc17a7", 
#              "Th", "Vip", "Tac1", "Pvalb", "Sst", "Ntrk1", "Tlx3", "Sim1", 
#              "Lypd6", "Snap25", "Ctgf", "Tacr1", "Slc6a4", "Pdyn", "Vipr2", 
#              "Rorb", "Oxtr", "Tnnt1", "Vipr2", "Slc17a8", "Cart", "Chat", "Npr3",
#              "Gad2", "Npr3", "Calb2", "Tcerg1L", "Slco2a1", "Rxfp1", "CMV", "Chodl",
#              "Necab1", "Penk1", "Adora2a", "Drd1a", "Camk2a", "Sert", "Drd2", "Otp", 
#              "Fos2A", "Htr2a", "Syt17", "Calcr", "Npas1", "Rasgrf2", "Anxa1", "CD4",
#              "Phox2b", "Ella", "Drd3", "CAG", "Aldh1"]

# ind = np.isin(cre_lines, global_bin_df['gene'])
# cre_filt_df = global_bin_df.iloc[np.isin(global_bin_df['gene'], cre_lines)]

# cre_lines = np.array(cre_lines)[ind]
# cre_lines = np.unique(cre_lines)

# %%

from ABC_toolbox import ABC_utils

all_gene = ABC_utils.load_gene("scRNAseq")

genes = ['Phox2b', 'Slc17a6', 'Slc6a5', 'Slc5a7', 'Alcam', 'Nxph1', 'Hoxb5', 'Tenm2', 
         'Meis2', 'Pcp4', 'Syt1', 'Rph3a', 'Zfhx3', 'Tshz2', 'Ebf3', 'Pax2', 'Arpp21', 
         'Robo1', 'Negr1', 'Rab3c']

from skimage.filters import threshold_otsu

# build list containing binary otsu threshold for every gene
n_genes = exp_boot.shape[1]
thresholds = []
for i in tqdm(range(n_genes)):
    thresholds.append(threshold_otsu(exp_boot[:,i]))
    
# binarize expression data according to previously calculated per-gene 
# otsu thresholds
exp_bin = exp_boot.copy()
for i in tqdm(range(n_genes)):
    exp_bin[:,i] = (exp_boot[:,i] > thresholds[i])
    
# create bool mask dict: dict where keys are cell types and values are bool 
# masks indicating which cells are that type
level = "cluster"
bool_mask_dict = {}
for label in np.unique(meta_boot[level]):
    bool_mask_dict[label] = (meta_boot[level] == label).values

weights = {}
for key in bool_mask_dict.keys():
    weights[key] = freqs[key]
    
weights = {k: v / total for total in (sum(weights.values()),) for k, v in weights.items()}

# calculate a given gene's global binariness
def calc_global_bin(exp, exp_raw, bool_mask_dict, weights):
    
    fracs = {}
    present = {}
    
    # iterate through cell types
    for key in bool_mask_dict.keys():
        
        # the number of cells of this type
        n_cells = sum(bool_mask_dict[key])
        
        # the fraction of cells of this type labeled as 1
        frac = sum(exp[bool_mask_dict[key]]) / n_cells
        
        fracs[key] = max((frac, 1-frac))
        
        if frac >= 0.5:
            present[key] = 1
        else:
            present[key] = 0
    
    fracs = np.array(list(fracs.values())) - 0.5
    fracs_mean = np.average(fracs, weights=list(weights.values()))
    
    global_bin = fracs_mean/0.5
    present_dict = present
    present = np.average(list(present.values()), weights=list(weights.values()))

    present_types = []
    off_types = []
    for key, value in present_dict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value == 1:
            present_types.append(key)
        elif value == 0:
            off_types.append(key)
            
    if len(present_types) != 0:
        avg_exp = []
        weights_present = []
        for present_type in present_types:
            weights_present.append(weights[present_type])
            avg_exp.append(np.mean(exp_raw[bool_mask_dict[present_type]]))
            
        avg_exp = np.average(avg_exp, weights=weights_present)
    else:
        avg_exp = 0
        
    if len(off_types) != 0:
        off_exp = []
        weights_off = []
        for off_type in off_types:
            weights_off.append(weights[off_type])
            off_exp.append(np.mean(exp_raw[bool_mask_dict[off_type]]))
            
        off_exp = np.average(off_exp, weights=weights_off)
    else:
        off_exp = 0
        
    avg_exp = avg_exp - off_exp
    avg_exp = max(0, avg_exp)
    
    return global_bin, present, avg_exp

global_bins = []
present_ratios = []
avg_exps = []
idx = []
for gene in tqdm(genes):
    i = (np.where(all_gene == gene)[0][0])
    idx.append(i)
    global_bin, present, avg_exp = calc_global_bin(exp_bin[:,i], exp_boot[:,i], bool_mask_dict, weights)
    global_bins.append(global_bin)
    present_ratios.append(present)
    avg_exps.append(avg_exp)
    
data_dict = {}
data_dict['gene'] = genes
data_dict['global_bin'] = global_bins
data_dict['present'] = present_ratios
data_dict['avg_exp'] = avg_exps
global_bin_df = pd.DataFrame(data=data_dict)

# %% allen cre lines


import matplotlib.pyplot as plt

test_gene = 'Gad2'

test_gene_ind = np.where(all_gene == test_gene)[0][0]

plt.hist(exp_super_raw[:,test_gene_ind], bins=100)

find_genes.plot_violin_gene(test_gene,
                            meta_super, 
                            exp_super_raw, 
                            level="subclass")

find_genes.plot_violin_gene(test_gene,
                            meta_super, 
                            exp_super_raw, 
                            level="cluster")

# %%

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

idx_genes = cre_lines

# expression matrix of good genes, where rows are genes and columns are observations
exp_good = exp_super_raw[:,idx].T

R = np.corrcoef(exp_good)
R, idx_genes = cluster_corr(R, idx_genes)

fig, ax = plt.subplots()
mappable = ax.imshow(R)

plt.gca().set_xticks(range(len(idx_genes)), idx_genes, rotation=45)
plt.gca().set_yticks(range(len(idx_genes)), idx_genes, rotation=45)
mappable.set_clim(-1, 1)
cbar = plt.colorbar(mappable)
cbar.set_label("PCC", rotation=270, labelpad=15)
