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

from ABC_toolbox import cell_funcs, gene_funcs, ABC_plot
# from utils import data_nav, find_genes, classify_cells, utils

f_gene = os.path.join(params.data_dir, "metadata\\WMB-10X\\20231215")
f_gene = os.path.join(f_gene, "gene.csv")
all_gene = pd.read_csv(f_gene)
all_gene = all_gene["gene_symbol"].values

# %% load ant-IRN-PARN data

# meta = pd.read_csv(os.path.join(params.local_data_dir, "MO-scRNAseq-meta.csv"), low_memory=False)
# exp = np.load(os.path.join(params.local_data_dir, "MO-scRNAseq-norm.npy"))
# freqs = pd.read_pickle(os.path.join(params.local_data_dir, "MO-MERFISH-freqs.pkl"))

# meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
# exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-norm.npy"))
# freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

meta = pd.read_csv(os.path.join(params.local_data_dir, "MRN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "MRN-scRNAseq-norm.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "MRN-MERFISH-freqs.pkl"))
subclass_freqs = cell_funcs.recalc_freqs(freqs, new_level='subclass')

# %%

inhib_kws = ["Vip Gaba", "Sncg Gaba", "Lamp5 Gaba", "Lamp5 Lhx6", 
             "Pvalb chandelier", "Pvalb Gaba", "Sst Gaba", "Sst Chodl"]

inhib_scale = 2

inhib_sum = 0
excit_sum = 0
for key, val in freqs.items():
    
    check_inhib = []
    for inhib_kw in inhib_kws:
        check_inhib.append(inhib_kw in key)
        
    if any(check_inhib):
        freqs[key] = inhib_scale * val
        inhib_sum += inhib_scale * val
    else:
        excit_sum += val
    
excit_target_sum = 1 - inhib_sum
s = excit_target_sum / excit_sum

for key, val in freqs.items():
    
    check_inhib = []
    for inhib_kw in inhib_kws:
        check_inhib.append(inhib_kw in key)
        
    if any(check_inhib):
        pass
    else:
        freqs[key] = s * val
        
# %%

def restrict_data(meta, exp, labels):
        
    bool_mask = np.zeros(len(meta), dtype=bool)
    
    # add cells to the bool mask present in labels at a given taxonomic level
    def _add_cells(level, bool_mask):
    
        meta_labels = meta[level].values
        
        for i, val in enumerate(meta_labels):
            if val in labels:
                bool_mask[i] = 1
                
        return bool_mask
    
    bool_mask = _add_cells("cluster", bool_mask)
    bool_mask = _add_cells("supertype", bool_mask)
    bool_mask = _add_cells("subclass", bool_mask)
    bool_mask = _add_cells("class", bool_mask)
    
    meta_idx = meta.index[bool_mask]
    meta_filt = meta.iloc[meta_idx]
    meta_filt.reset_index(drop=True, inplace=True)
    
    exp_filt = exp[bool_mask, :]

    return meta_filt, exp_filt

    

# %%

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=3)
# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=20000)


# %%

exp_boot = exp_boot + 1
exp_boot = np.log2(exp_boot)

# %%

excit = ['01 IT-ET Glut', '02 NP-CT-L6b Glut']
inhib = ['06 CTX-CGE GABA', '07 CTX-MGE GABA', '08 CNU-MGE GABA']
no_vip = ['047 Sncg Gaba', '049 Lamp5 Gaba','051 Pvalb chandelier Gaba', 
          '052 Pvalb Gaba', '053 Sst Gaba', '056 Sst Chodl Gaba']
no_lamp = ['047 Sncg Gaba','051 Pvalb chandelier Gaba', 
          '052 Pvalb Gaba', '053 Sst Gaba', '056 Sst Chodl Gaba']

meta_boot, exp_boot = restrict_data(meta_boot, exp_boot, excit) 

meta_boot, exp_boot = restrict_data(meta_boot, exp_boot, inhib) 
meta_boot, exp_boot = restrict_data(meta_boot, exp_boot, no_vip) 
meta_boot, exp_boot = restrict_data(meta_boot, exp_boot, no_lamp)

# %%

excit = ['154 PF Fzd5 Glut', '155 PRC-PAG Pax6 Glut',
       '156 MB-ant-ve Dmrta2 Glut', '157 RN Spp1 Glut',
       '158 MRN-PAG Nkx6-1 Glut', '160 PAG-SC Neurod2 Meis2 Glut',
       '162 CUN Evx2 Lhx2 Glut', '163 APN C1ql2 Glut',
       '164 APN C1ql4 Glut', '165 PAG-MRN Pou3f1 Glut',
       '166 MRN Pou3f1 C1ql4 Glut', '167 PRC-PAG Tcf7l2 Irx2 Glut',
       '168 SPA-SPFm-SPFp-POL-PIL-PoT Sp9 Glut',
       '169 PAG-SC Pou4f1 Zic1 Glut', '170 PAG-MRN Tfap2b Glut',
       '171 PAG Pou4f1 Bnc2 Glut', '172 PAG Pou4f1 Ebf2 Glut',
       '180 SCiw Pitx2 Glut', '181 IC Tfap2d Maf Glut',
       '182 CUN-PPN Evx2 Meis2 Glut', '190 ND-INC Foxd2 Glut','221 LDT-PCG Vsx2 Lhx4 Glut',
       '222 PB Evx2 Glut', '224 PCG-PRNr Vsx2 Nkx6-1 Glut',
       '243 PGRN-PARN-MDRN Hoxb5 Glut', '250 CBN Neurod2 Pvalb Glut']
inhib = ['191 PAG-MRN Rln3 Gaba', '193 MRN-PPN-CUN Pax8 Gaba',
         '194 MRN-VTN-PPN Pax5 Cdh23 Gaba', '195 SNr-VTA Pax5 Npas1 Gaba',
         '196 PAG-PPN Pax5 Sox21 Gaba', '197 SNr Six3 Gaba',
         '198 IC Six3 En2 Gaba', '199 PAG-MRN-RN Foxa2 Gaba',
         '200 PAG-ND-PCG Onecut1 Gaba', '201 PAG-RN Nkx2-2 Otx1 Gaba',
         '202 PRT Tcf7l2 Gaba', '203 LGv-SPFp-SPFm Nkx2-2 Tcf7l2 Gaba',
         '205 SC-PAG Lef1 Emx2 Gaba', '206 SCm-PAG Cdh23 Gaba',
         '207 SCs Dmbx1 Gaba', '208 SC Lef1 Otx2 Gaba',
         '209 SCs Pax7 Nfia Gaba','263 CS-RPO Meis2 Gaba',
         '264 PRNc Otp Gly-Gaba', '267 CS-PRNr-PCG Tmem163 Otp Gaba',
         '268 CS-PRNr-DR En1 Sox2 Gaba', '270 LDT-DTN Gata3 Nfix Gaba',
         '272 LDT-PCG-CS Gata3 Lhx1 Gaba', '277 DTN-LDT-IPN Otp Pax3 Gaba',
         '278 NLL Gata3 Gly-Gaba', '285 MY Lhx1 Gly-Gaba']

# meta_boot, exp_boot = restrict_data(meta_boot, exp_boot, excit)
meta_boot, exp_boot = restrict_data(meta_boot, exp_boot, inhib) 

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

# genes = ['Phox2b', 'Slc17a6', 'Slc6a5', 'Slc5a7', 'Alcam', 'Nxph1', 'Hoxb5', 'Tenm2', 
#          'Meis2', 'Pcp4', 'Syt1', 'Rph3a', 'Zfhx3', 'Tshz2', 'Ebf3', 'Pax2', 'Arpp21', 
#          'Robo1', 'Negr1', 'Rab3c']

genes = all_gene

# genes = np.load("slc6a5_corr_genes.npy", allow_pickle=True)

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

# normalize cell types weights to sum to 1
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

# %%

global_bin_df_filt = global_bin_df.copy()
global_bin_df_filt = global_bin_df_filt.query("present != 0")
global_bin_df_filt = global_bin_df_filt.query("present != 1")
global_bin_df_filt = global_bin_df_filt.query("global_bin > 0.5")
global_bin_df_filt = global_bin_df_filt.query("present < 0.8 & present > 0.2")
global_bin_df_filt.reset_index(inplace=True)

best_genes = global_bin_df_filt['gene'].values

# %% 

global_bin_df_filt = global_bin_df

# %%

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots()

order = np.argsort(global_bin_df_filt['avg_exp'].values)
plt.scatter(global_bin_df_filt['present'].values[order], 
            global_bin_df_filt['global_bin'].values[order], 
            c=global_bin_df_filt['avg_exp'].values[order], 
            vmin=0,
            cmap="plasma")

plt.xlabel("Weighted Presence Ratio")
plt.ylabel("Weighted Local Binariness")

plt.xlim(0, 1)
plt.ylim(0, 1)

from matplotlib.patches import Arc

d = 0.4
ax.add_patch(Arc((0.5, 1), d, d, ls='--'))
ax.set_aspect('equal')
cbar = plt.colorbar()
cbar.set_label("Log-fold on-off expression difference", rotation=270, labelpad=20)

offset = 0.005
best_genes = []
for i in range(len(global_bin_df_filt)):
    x = global_bin_df_filt['present'][i]
    y = global_bin_df_filt['global_bin'][i]
    if np.sqrt((x - 0.5)**2 + (y - 1)**2) < d/2:
        plt.annotate(global_bin_df_filt['gene'][i], (x + offset, y + offset))
        best_genes.append(global_bin_df_filt['gene'][i])
        
plt.gca().set_aspect('equal')

# %% allen cre lines

import matplotlib.pyplot as plt

test_gene = 'Six3'

# test_gene_ind = np.where(all_gene == test_gene)[0][0]

# plt.hist(exp_super[:,test_gene_ind], bins=100)

ABC_plot.plot_gene(test_gene,
                   meta_boot, 
                   exp_boot, 
                   level="subclass")

ABC_plot.plot_gene(test_gene,
                   meta_boot, 
                   exp_boot, 
                   level="cluster")


# %%

import matplotlib.pyplot as plt
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

best_genes = np.array(best_genes)

gene_idx = []
for gene in best_genes:
    gene_idx.append(np.where(all_gene == gene)[0][0])

# expression matrix of good genes, where rows are genes and columns are observations
exp_good = exp_boot[:,gene_idx].T

R = np.corrcoef(exp_good)
R, idx_genes = cluster_corr(R, best_genes)

fig, ax = plt.subplots()
mappable = ax.imshow(R)

plt.gca().set_xticks(range(len(idx_genes)), idx_genes, rotation=45)
plt.gca().set_yticks(range(len(idx_genes)), idx_genes, rotation=45)
mappable.set_clim(-1, 1)
cbar = plt.colorbar(mappable)
cbar.set_label("PCC", rotation=270, labelpad=15)
