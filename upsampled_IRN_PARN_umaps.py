# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:17:14 2024

@author: jpv88
"""

import scvi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData, concat, read_h5ad
from scipy import stats
from tqdm import tqdm

import math
import os
import random
import sys

# from utils import utils
from ABC_toolbox import ABC_utils, ABC_plot, cell_funcs

# obtain this file's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

sys.path.append(dname)
import params

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp_raw = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
ratios = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

# supercells is a bad idea here, it makes the cell types too different, every cluster
# forms its own island
# exp_raw, meta = cell_funcs.boot_super(exp_raw, meta, k=5)

genes = ABC_utils.load_gene("scRNAseq")

adata = AnnData(exp_raw)

# assign batch
n_ABC = adata.shape[0]
adata.obs['batch'] = n_ABC*['ABC']

# assign cell type
adata.obs["cluster"] = meta["cluster"].values
adata.obs["supertype"] = meta["supertype"].values
adata.obs["subclass"] = meta["subclass"].values
adata.obs["class"] = meta["class"].values

# assign gene names
adata.var_names = genes


# %%

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")

model = scvi.model.SCVI(
    adata,
    n_layers=2,
    n_latent=40
)

model.train(max_epochs=100,
            early_stopping=True,
            early_stopping_patience=5)


# %%

scanvi_model = scvi.model.SCANVI.from_scvi_model(
    model,
    adata=adata,
    labels_key="cluster",
    unlabeled_category="Unknown",
)

scanvi_model.train(max_epochs=50,
                   early_stopping=True,
                   early_stopping_patience=5)

# %%

tot_cells = 20000       # total number of cells desired
n_cells = {}            # cells needed of each type to match that target
avail_cells = {}        # cells available of each type in raw data
needed_rolls = {}
needed_cells = {}

# iterate through ABC clusters
for key in ratios.keys():
    n_cells[key] = round(ratios[key]*tot_cells)                       # how many cells of this cluster do we need
    avail_cells[key] = sum(meta["cluster"] == key)                    # how many actual cells of this cluster exist
    needed_cells[key] = np.max([0, n_cells[key] - avail_cells[key]])
    needed_rolls[key] = math.ceil(needed_cells[key]/avail_cells[key])
    
max_needed_rolls = max(needed_rolls.values())

# %% sample new cells

clu = meta["cluster"].values

new_cells = {}
for key in n_cells.keys():
    new_cells[key] = []

for i in tqdm(range(max_needed_rolls)):
    
    samples = scanvi_model.posterior_predictive_sample()
    samples = samples.to_scipy_sparse()
    samples = samples.todense()
    
    for key in needed_cells.keys():
        
        if needed_cells[key] > 0:
            
            clu_mask = (clu == key)
            exp_clu = samples[clu_mask, :]
            
            cells_this_roll = exp_clu.shape[0]
            to_insert = exp_clu[:np.min([cells_this_roll, needed_cells[key]]), :]
            
            new_cells[key].append(to_insert)
            
            needed_cells[key] -= to_insert.shape[0]
            
for k, v in new_cells.items():
    
    if len(v) != 0:
        
        new_cells[k] = np.concatenate(v)
        
c_type = []
all_new_cells = []

for k, v in new_cells.items():
    
    if len(v) != 0:
        
        all_new_cells.append(v)
        c_type.append(np.repeat(k, v.shape[0]))

c_type = np.concatenate(c_type)
all_new_cells = np.concatenate(all_new_cells)

new_adata = AnnData(all_new_cells)

# assign batch
n_ABC = new_adata.shape[0]
new_adata.obs['batch'] = n_ABC*['ABC']

# assign cell type
new_adata.obs["cluster"] = c_type

# assign gene names
new_adata.var_names = adata.var_names

del adata.obs["supertype"]
del adata.obs["subclass"]
del adata.obs["class"]
del adata.obs["_scvi_batch"]
del adata.obs["_scvi_labels"]
del adata.uns["_scvi_uuid"]
del adata.uns["_scvi_manager_uuid"]

adata.var_names_make_unique()
new_adata.var_names_make_unique()

adata_us = concat([adata, new_adata])

# %%

# iterate through ABC clusters
# k = ABC cluster
# v = number of cells of that cluster needed
for k, v in n_cells.items():
    
    adata_us_tot_cells = len(adata_us)
    mask = (adata_us.obs["cluster"] == k)
    mask = mask.values
    adata_us_cells = sum(mask)
    
    if adata_us_cells < v:
        print("problem")
        
    elif adata_us_cells == v: 
        pass
    
    elif adata_us_cells > v:
        cells_to_del = (adata_us_cells - v)
        idx = np.arange(adata_us_tot_cells)[mask]
        idx_to_del = random.sample(list(idx), cells_to_del)
        mask = np.ones(adata_us_tot_cells, dtype=bool)
        mask[idx_to_del] = False
        adata_us = adata_us[mask,:]

checks = []
for k, v in n_cells.items():
    
    mask = (adata_us.obs["cluster"] == k)
    mask = mask.values
    adata_us_cells = sum(mask)
    checks.append((adata_us_cells, v))
    
supertype = []
subclass = []
_class = []

for ct in adata_us.obs["cluster"]:
    ctax = ABC_utils.fetch_cluster_tax(ct)
    supertype.append(str(ctax[0]))
    subclass.append(str(ctax[1]))
    _class.append(str(ctax[2]))
    
adata_us.obs["supertype"] = supertype
adata_us.obs["subclass"] = subclass
adata_us.obs["class"] = _class

    
# %%

sc.pp.normalize_total(adata_us)
# adata_us.X = np.round(adata_us.X).astype(np.uint16)

sc.pp.highly_variable_genes(
    adata_us,
    flavor="seurat_v3",
    n_top_genes=2000
)

# adata_us.X = np.round(adata_us.X).astype(np.uint16)

# %%

adata_us_copy = sc.pp.log1p(adata_us, copy=True)
sc.pp.scale(adata_us_copy)
sc.pp.pca(adata_us_copy)
sc.pp.neighbors(adata_us_copy)
sc.tl.umap(adata_us_copy)
adata_us.obsm['X_umap'] = adata_us_copy.obsm['X_umap']

# %%

adata_us.write_h5ad("antIRN-PARN-upsampled")

# %%

import matplotlib as mpl

level = "subclass"

sc.pl.umap(adata_us, 
           color=level, 
           s=60, 
           palette=ABC_plot.fetch_colors_dict(level),
           sort_order=False) 

cell_types = np.unique(adata_us.obs[level])

for ct in cell_types:
    mask = (adata_us.obs[level] == ct)
    umap_ct = adata_us.obsm["X_umap"][mask,:]
    cent = np.mean(umap_ct, axis=0)
    plt.annotate(ct, cent, ha="center", fontsize=8)
    
mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()
    
# %%

adata_us.obsm["X_pca_MDE"] = scvi.model.utils.mde(adata_us.obsm["X_pca"])

# %%

level = "subclass"

sc.pl.embedding(
    adata_us,
    basis="X_pca_MDE",
    color=level,
    frameon=False,
    ncols=1,
    s=50,
    palette=utils.fetch_colors_dict(level)
    
)

cell_types = np.unique(adata_us.obs[level])

for ct in cell_types:
    mask = (adata_us.obs[level] == ct)
    umap_ct = adata_us.obsm["X_pca_MDE"][mask,:]
    cent = np.mean(umap_ct, axis=0)
    plt.annotate(ct, cent, ha="center", fontsize=6)
