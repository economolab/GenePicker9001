# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:24:17 2025

@author: jpv88
"""

import logging
import os
import pygad
import random

import numpy as np
import pandas as pd

import params

from ABC_toolbox import cell_funcs, classify_cells, gene_funcs
from gene_filter import filt_genes
from genetic_algo import run_ga

# %% params

# number of genes to keep after filtering using heuristics
top_n = 500

# %% run ga

gene_df_gas = []
sol_df_gas = []
ga_instances = []

for _ in range(20):

    meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
    exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
    freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
    
    exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)
    
    exp_super = gene_funcs.normalize_counts_to_median(exp_super)
    
    
    gene_df, top_genes = filt_genes(exp_super, meta_super, freqs, top_n=top_n)
    
    # bootstrap distribution from scRNAseq that matches MERFISH frequencies
    exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                        n=5000)
    
    exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)
    
    gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes, 
                                                num_generations=100, 
                                                mutation_probability=[0.5, 0])
    
    gene_df_gas.append(gene_df_ga)
    sol_df_gas.append(sol_df_ga)
    ga_instances.append(ga_instance)

# %%

def jaccard_distance(A, B):
    
    A = set(A)
    B = set(B)
    
    num = len(A.symmetric_difference(B))
    denom = len(A.union(B))
    
    d = num / denom
    
    return d

from sklearn.metrics import pairwise_distances

solutions = sol_df_ga["solution"].values

D = pairwise_distances(np.vstack(solutions), metric=jaccard_distance)

# %%

from sklearn.manifold import MDS

embedding = MDS(n_components=2, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(D)


# %%

import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots()

ax.scatter(X_transformed[:,0], X_transformed[:,1], c=sol_df_ga['fitness'].values)
# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

tri = mpl.tri.Triangulation(X_transformed[:,0], X_transformed[:,1])
tri = mpl.tri.CubicTriInterpolator(tri, sol_df_ga['fitness'].values)

ax.plot_trisurf(X_transformed[:,0], X_transformed[:,1], 
                tri(X_transformed[:,0], X_transformed[:,1]),
                cmap=plt.cm.CMRmap)

ax.plot_trisurf(X_transformed[:,0], X_transformed[:,1], sol_df_ga['fitness'].values,
                cmap=plt.cm.CMRmap)


