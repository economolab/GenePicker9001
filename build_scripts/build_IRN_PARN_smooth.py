# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:48:38 2024

@author: jpv88
"""

import numpy as np
import pandas as pd
import scipy as sp

import scanpy as sc
from anndata import AnnData, read_h5ad

from utils import data_nav, find_genes, classify_cells, utils

import os
import sys

# obtain this file's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

sys.path.append(dname)
import params

f_gene = os.path.join(params.data_dir, "metadata\\WMB-10X\\20231215")
f_gene = os.path.join(f_gene, "gene.csv")
all_gene = pd.read_csv(f_gene)
all_gene = all_gene["gene_symbol"].values

# %%

roi = ['IRN', 'PARN']
level = 'structure'

MERFISH_meta = data_nav.load_meta('MERFISH')
scRNAseq_meta = data_nav.load_meta('scRNAseq')

MERFISH_meta = data_nav.extract_MERFISH_meta(MERFISH_meta, 
                                              roi, 
                                              kind='restrict',
                                              category='anatomy',
                                              level=level)

non_neuronal = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']

MERFISH_meta = data_nav.extract_MERFISH_meta(MERFISH_meta, 
                                              non_neuronal, 
                                              kind='remove',
                                              category='taxonomy',
                                              level='class')

MERFISH_meta.drop(columns=["Unnamed: 0"], inplace=True)

# %%

minimum_cells = 3

clusters = np.unique(MERFISH_meta['cluster'].values)
bool_mask = np.ones(len(MERFISH_meta), dtype=bool)

for clu in clusters:
    num = sum(MERFISH_meta["cluster"].values == clu)
    if num < minimum_cells:
        idx = np.where(MERFISH_meta["cluster"].values == clu)[0]
        bool_mask[idx] = 0
    
MERFISH_meta = MERFISH_meta.iloc[MERFISH_meta.index[bool_mask]]
MERFISH_meta.reset_index(drop=True, inplace=True)

d = data_nav.calc_frac_per_clu(MERFISH_meta, level="cluster")

# %%

import matplotlib.pyplot as plt

level = 'subclass'
clusters = np.unique(MERFISH_meta[level].values)
colors_dict = utils.fetch_colors_dict(level)
colors = [colors_dict[el] for el in clusters]

min_x = np.floor(np.min(MERFISH_meta["x_ccf"]) * 10) / 10
max_x = np.ceil(np.max(MERFISH_meta["x_ccf"]) * 10) / 10

step = 0.1
x_range = np.arange(min_x, max_x + step, step)

x_ct_dict = {}
for val in x_range:
    x_ct_dict[np.round(val, decimals=1)] = []
    
for i in range(len(MERFISH_meta)):
    key = np.round(MERFISH_meta.iloc[i]['x_ccf'], decimals=1)
    x_ct_dict[key].append(MERFISH_meta.iloc[i][level])

freqs = np.zeros((len(clusters), len(x_range)), dtype=np.uint32)

clusters = np.unique(MERFISH_meta[level].values)

for i, clu in enumerate(clusters):
    for j, val in enumerate(list(x_ct_dict.values())):
        freqs[i,j] += sum(np.array(val) == clu)
        
freqs = freqs/freqs.sum(axis=0, keepdims=1)


# %%

from scipy.interpolate import CubicSpline, make_smoothing_spline

lam = 0.001

new_x = np.linspace(min_x, max_x, 1000)

freqs_smooth = np.zeros((len(clusters), len(new_x)))
for i in range(freqs.shape[0]):
    spl = make_smoothing_spline(x_range, freqs[i,:], lam=lam)
    freqs_smooth[i,:] = spl(new_x)
    
freqs_smooth[freqs_smooth < 0] = 0
freqs_smooth = freqs_smooth/freqs_smooth.sum(axis=0, keepdims=1)
    
plt.stackplot(new_x, freqs_smooth, colors=colors, baseline='zero', labels=clusters)
plt.xlabel("<---- Anterior | Posterior ---->")
plt.ylabel("Proportion")
plt.title("Cell types in IRN/PARN (subclass-level)")

# %%

adata = AnnData(freqs_smooth)
adata.obs_names = clusters

d = {}
d['x_ccf'] = new_x
df = pd.DataFrame(data=d)
adata.var = df

adata.write_h5ad('IRN_PARN_all_ratios')

# %%

ratios = utils.find_ratios_IRN_PARN(11.2)

# %%

import pickle

clusters = np.unique(MERFISH_meta["cluster"].values)
scRNAseq_raw = data_nav.fetch_scRNAseq(clusters, scRNAseq_meta, form='raw')

exp = scRNAseq_raw[1]
meta = scRNAseq_raw[0]
exp = exp.todense()

meta.to_csv('IRN_PARN_meta_pp.csv')

np.save("IRN_PARN_raw", exp)


# %%

import pickle

clusters = np.unique(MERFISH_meta["cluster"].values)
scRNAseq_raw = data_nav.fetch_scRNAseq(clusters, scRNAseq_meta, form='raw')
scRNAseq_log2 = data_nav.fetch_scRNAseq(clusters, scRNAseq_meta, form='log2')

exp = scRNAseq_raw[1]
meta = scRNAseq_raw[0]
exp_log2 = scRNAseq_log2[1] 
exp = exp.todense()

adata = AnnData(exp)
adata.var_names = all_gene

adata_all_genes = adata.copy()
adata.write_h5ad('IRN_PARN_pp_all_genes')

sc.pp.recipe_zheng17(adata)

adata.write_h5ad('IRN_PARN_pp')

meta.to_csv('IRN_PARN_meta_pp.csv')

np.save("IRN_PARN_log2", exp_log2.todense())
np.save("IRN_PARN_raw", exp)

d = data_nav.calc_frac_per_clu(MERFISH_meta, level="cluster")

with open('IRN_PARN_ratios.pkl', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

