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

from ABC_toolbox import ABC_plot, ABC_utils, cell_funcs, classify_cells, gene_funcs, iterative_reclustering
from gene_filter import filt_genes
from genetic_algo import run_ga

# %% params

# number of genes to keep after filtering using heuristics
top_n = 500

# %% run ga

gene_df_gas_500 = []
sol_df_gas_500 = []
ga_instances_500 = []

for _ in range(10):

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
    
    gene_df_gas_500.append(gene_df_ga)
    sol_df_gas_500.append(sol_df_ga)
    ga_instances_500.append(ga_instance)

tested_genes = []
for gene_df in gene_df_gas_500:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)

final_gene_scores = []
for gene in tested_genes:
    scores = []
    occurences = []
    for gene_df in gene_df_gas_500:
        if gene in gene_df['gene'].values:
            scores.append(gene_df['fitnesses'][gene_df['gene'] == gene].values[0])
            occurences.append(gene_df['occurences'][gene_df['gene'] == gene].values[0])
    final_gene_scores.append(np.average(scores, weights=occurences))
    
final_gene_df_500 = {}
final_gene_df_500['gene'] = tested_genes
final_gene_df_500['score'] = final_gene_scores
final_gene_df_500 = pd.DataFrame(data=final_gene_df_500)

final_gene_df_filt = final_gene_df_500.nlargest(200, 'score')

gene_df_gas_200 = []
sol_df_gas_200 = []
ga_instances_200 = []

top_genes = final_gene_df_filt['gene'].values

for _ in range(10):

    meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
    exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
    freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
    
    exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)
    
    exp_super = gene_funcs.normalize_counts_to_median(exp_super)
    
    # bootstrap distribution from scRNAseq that matches MERFISH frequencies
    exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                        n=5000)
    
    exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)
    
    gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes, 
                                                init_copies=4,
                                                num_generations=100, 
                                                mutation_probability=[0.5, 0])
    
    gene_df_gas_200.append(gene_df_ga)
    sol_df_gas_200.append(sol_df_ga)
    ga_instances_200.append(ga_instance)
    
# %% collate results

tested_genes = []
for gene_df in gene_df_gas_200:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)

final_gene_scores = []
for gene in tested_genes:
    scores = []
    occurences = []
    for gene_df in gene_df_gas_200:
        if gene in gene_df['gene'].values:
            scores.append(gene_df['fitnesses'][gene_df['gene'] == gene].values[0])
            occurences.append(gene_df['occurences'][gene_df['gene'] == gene].values[0])
    final_gene_scores.append(np.average(scores, weights=occurences))
    
final_gene_df = {}
final_gene_df['gene'] = tested_genes
final_gene_df['score'] = final_gene_scores
final_gene_df = pd.DataFrame(data=final_gene_df)

# %%

for i, ga in enumerate(ga_instances_500):
    ga.save(f"ga_500_{i}")
    
for i, df in enumerate(gene_df_gas_500):
    df.to_csv(f"gene_df_500_{i}.csv")
    
for i, df in enumerate(sol_df_gas_500):
    df.to_csv(f"sol_df_500_{i}.csv")
    
final_gene_df_500.to_csv("final_gene_df_500.csv")

for i, ga in enumerate(ga_instances_200):
    ga.save(f"ga_200_{i}")
    
for i, df in enumerate(gene_df_gas_200):
    df.to_csv(f"gene_df_200_{i}.csv")
    
for i, df in enumerate(sol_df_gas_200):
    df.to_csv(f"sol_df_200_{i}.csv")
    
final_gene_df.to_csv("final_gene_df_200.csv")


# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)

exp_super = gene_funcs.normalize_counts_to_median(exp_super)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=5000)



# %%

dir1 = r"C:\Users\jpv88\Documents\GenePicker9001_data\2025-03-30 GA run"
dir2 = r"C:\Users\jpv88\Documents\GenePicker9001_data\2025-03-31 GA run"
dir3 = r"C:\Users\jpv88\Documents\GenePicker9001_data\2025-04-01 GA run"

gene_df_1 = pd.read_csv(os.path.join(dir1, "final_gene_df_200.csv"))
gene_df_2 = pd.read_csv(os.path.join(dir2, "final_gene_df_200.csv"))
gene_df_3 = pd.read_csv(os.path.join(dir2, "final_gene_df_200.csv"))

gene_dfs = [gene_df_1, gene_df_2, gene_df_3]

tested_genes = []
for gene_df in gene_dfs:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)

final_gene_scores = []
for gene in tested_genes:
    scores = []
    occurences = []
    for gene_df in gene_dfs:
        if gene in gene_df['gene'].values:
            scores.append(gene_df['score'][gene_df['gene'] == gene].values[0])
            occurences.append(1)
    final_gene_scores.append(np.average(scores, weights=occurences))
    
final_gene_df = {}
final_gene_df['gene'] = tested_genes
final_gene_df['score'] = final_gene_scores
final_gene_df = pd.DataFrame(data=final_gene_df)

# %%

final_gene_df_filt = final_gene_df.nlargest(100, 'score')

gene_df_gas_100 = []
sol_df_gas_100 = []
ga_instances_100 = []

top_genes = final_gene_df_filt['gene'].values

for _ in range(10):

    meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
    exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
    freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
    
    exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)
    
    exp_super = gene_funcs.normalize_counts_to_median(exp_super)
    
    # bootstrap distribution from scRNAseq that matches MERFISH frequencies
    exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                        n=5000)
    
    exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)
    
    gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes, 
                                                init_copies=6,
                                                num_generations=100, 
                                                mutation_probability=[0.5, 0])
    
    gene_df_gas_100.append(gene_df_ga)
    sol_df_gas_100.append(sol_df_ga)
    ga_instances_100.append(ga_instance)
    
# %% collate results

tested_genes = []
for gene_df in gene_df_gas_100:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)

final_gene_scores = []
for gene in tested_genes:
    scores = []
    occurences = []
    for gene_df in gene_df_gas_100:
        if gene in gene_df['gene'].values:
            scores.append(gene_df['fitnesses'][gene_df['gene'] == gene].values[0])
            occurences.append(gene_df['occurences'][gene_df['gene'] == gene].values[0])
    final_gene_scores.append(np.average(scores, weights=occurences))
    
final_gene_df = {}
final_gene_df['gene'] = tested_genes
final_gene_df['score'] = final_gene_scores
final_gene_df = pd.DataFrame(data=final_gene_df)
    
# %%

for i, ga in enumerate(ga_instances_100):
    ga.save(f"ga_100_{i}")
    
for i, df in enumerate(gene_df_gas_100):
    df.to_csv(f"gene_df_100_{i}.csv")
    
for i, df in enumerate(sol_df_gas_100):
    df.to_csv(f"sol_df_100_{i}.csv")
    
final_gene_df.to_csv("final_gene_df_100.csv")
    
# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)

exp_super = gene_funcs.normalize_counts_to_median(exp_super)

gene_df, top_genes = filt_genes(exp_super, meta_super, freqs, top_n=top_n)

weights = [(1/3), (2/3)]

arr = np.column_stack((gene_df["rdr"], gene_df["var_E"]))
scores = np.average(arr, axis=1, weights=weights)

gene_df["scores"] = scores
gene_df = gene_df.sort_values('scores', ascending=False)
gene_df.reset_index(drop=True, inplace=True)

# %%

import matplotlib.pyplot as plt

inds = []
for gene in final_gene_df['gene'].values:
    inds.append(np.where(gene_df['gene'].values == gene)[0][0])

plt.hist(inds, bins=30)
plt.xlim(0, 500)
plt.xlabel("Heuristic rank")
plt.ylabel("Frequency")
plt.title("Initial heuristic rank of top 100 final genes")

# %%

ABC_plot.plot_gene("Snap25", meta_boot, exp_boot, level='cluster')

# %%

dir_4 = r"C:\Users\jpv88\Documents\GenePicker9001_data\2025-04-01 GA run final 100"

gene_df_4 = pd.read_csv(os.path.join(dir_4, "final_gene_df_100.csv"))

genes = gene_df_4['gene'].values
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

genes = ['Tenm2', 'Cacna2d1', 'Ebf3', 'Syt2', 'Hcn1', 'Nxph1', 'Zfhx3', 'Kcnip4', 'Hoxb5', 'Nrxn3', 'Pbx3', 'Pcdh9']
all_genes = ABC_utils.load_gene("scRNAseq")

gene_idx = []
for gene in genes:
    gene_idx.append(np.where(all_genes == gene)[0][0])

final_labels, final_cm, all_accs, all_n_labels = iterative_reclustering.iterative_recluster(meta_super, 
                                                                                            exp_super[:,gene_idx], 
                                                                                            freqs)
# %%

fig, ax = plt.subplots()

ax.plot(all_n_labels, all_accs)
plt.xlabel("# cluster groups")
plt.ylabel("Accuracy")
plt.title("Optimized panel, scRNAseq cross-validation of KNN classifier (99% acc at 17 cgs)")

# %%

def build_clu_mapping(labels, labels_mix):
    
    clu_mapping = {}
    
    for label in labels:
        for label_list in labels_mix:
            label_list = label_list.split(", ")
            if label in label_list:
                clu_mapping[label] = label_list
                break
    
    for k in clu_mapping.keys():
        clu_mapping[k] = str(clu_mapping[k])
        clu_mapping[k] = clu_mapping[k].translate({ord(c): None for c in "'[]"})
        
    keys = list(clu_mapping.keys())
    for k in keys:
        if ',' in k:
            k_split = k.split(", ")
            for k2 in k_split:
                clu_mapping[k2] = clu_mapping[k]
    
    return clu_mapping

def recalc_ratios(ratios, labels):
    
    uniq_clu = list(ratios.keys())
    clu_mapping = build_clu_mapping(uniq_clu, labels)
    
    new_ratios = {}
    for label in labels:
        new_ratios[label] = 0
        
    for clu in uniq_clu:
        new_ratios[clu_mapping[clu]] += ratios[clu]
        
    return new_ratios

# %%

new_freqs = recalc_ratios(freqs, final_labels)



