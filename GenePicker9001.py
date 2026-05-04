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

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from ABC_toolbox import ABC_plot, ABC_utils, cell_funcs, classify_cells, gene_funcs, cluster_recombination
from gene_filter import filt_genes
from genetic_algo import run_ga, find_genes

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

# %%

mix_targets = []

# motor neurons
mix_targets.append(('4601 HB Calcb Chol_1', '4603 HB Calcb Chol_3', '4604 HB Calcb Chol_4', '4605 HB Calcb Chol_5',
                    '4553 DMX VII Tbx20 Chol_1', '4554 DMX VII Tbx20 Chol_1', '4555 DMX VII Tbx20 Chol_1'))

# mossy fibers
mix_targets.append(('4198 PG-TRN-LRN Fat2 Glut_1', '4199 PG-TRN-LRN Fat2 Glut_1'))

labels = list(freqs.keys())
new_groups = cluster_recombination.manual_recluster(labels, mix_targets)
labels_mix = cluster_recombination.mix_labels(labels, new_groups)
clu_mapping = cluster_recombination.build_clu_mapping(labels, labels_mix)

# %%

find_genes(meta, exp, freqs, num_generations=100, num_iter=10, filt_order=[1000], init_copies=2,
           rdr_N=10, var_E_N=2, skip_heur=False, clu_mapping=clu_mapping)

# %%

# def t(meta, exp, freqs):
    
#     with Pool(processes=2) as pool:

#         kwds = {'filt_order': [1000], 'num_iter': 1, 'num_generations': 1, 'rdr_N': 1, 'var_E_N': 1, 'skip_heur': True}
#         multiple_results = [pool.apply_async(find_genes, args=(meta, exp, freqs), kwds=kwds) for i in range(2)]
#         # res = pool.apply_async(find_genes, args=(meta, exp, freqs), kwds=kwds)
#         # res.get(timeout=None)
#         print([res.get(timeout=None) for res in multiple_results])
        
if __name__ == '__main__':
    
    with Pool(processes=2) as pool:

        kwds = dict(filt_order=[100], num_iter=1, num_generations=1, rdr_N=1, var_E_N=1, skip_heur=True, 
                init_copies=1)
        multiple_results = [pool.apply_async(find_genes, (meta, exp, freqs), kwds) for i in range(2)]
        
        pool.close()
        pool.join()
        
        # res = pool.apply_async(find_genes, args=(meta, exp, freqs), kwds=kwds)
        # res.get(timeout=None)
        print([res.get(timeout=None) for res in multiple_results])
        
        


# %% params

# number of genes to keep after filtering using heuristics
top_n = 1000

# number of genes in panel
num_genes = 24

# number of generations, default = 100
num_generations = 100

# number of guaranteed copies of each gene in starter population, default = 2
init_copies = 2

# %%

gene_df_gas_1000 = []
sol_df_gas_1000 = []
ga_instances_1000 = []

for _ in range(10):

    meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
    exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
    freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
    
    exp = gene_funcs.normalize_counts_to_median(exp)
    
    exp_super, meta_super = cell_funcs.boot_super(exp, meta)
    gene_df, top_genes = filt_genes(exp_super, meta_super, freqs, top_n=top_n)
    
    splits = cell_funcs.K_fold_cells(meta)
    supers = cell_funcs.boot_super_splits(exp, meta, splits, k=3)
    
    # bootstrap distribution from scRNAseq that matches MERFISH frequencies
    boots = cell_funcs.bootstrap_scRNAseq_splits(supers, freqs, n=5000)
    
    exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)
    
    gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes,
                                                num_genes=num_genes,
                                                num_generations=num_generations,
                                                init_copies=init_copies,
                                                mutation_probability=[0.5, 0],
                                                set_genes=['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7',
                                                           'Hoxb5', 'Ebf3', 'Pbx3', 'Nxph1',
                                                           'Arpp21', 'Zfhx4', 'Meis2', 'Rbms3'])
    
    gene_df_gas_1000.append(gene_df_ga)
    sol_df_gas_1000.append(sol_df_ga)
    ga_instances_1000.append(ga_instance)

tested_genes = []
for gene_df in gene_df_gas_1000:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)

final_gene_scores = []
for gene in tested_genes:
    scores = []
    occurences = []
    for gene_df in gene_df_gas_1000:
        if gene in gene_df['gene'].values:
            scores.append(gene_df['fitnesses'][gene_df['gene'] == gene].values[0])
            occurences.append(gene_df['occurences'][gene_df['gene'] == gene].values[0])
    final_gene_scores.append(np.average(scores, weights=occurences))
    
final_gene_df_1000 = {}
final_gene_df_1000['gene'] = tested_genes
final_gene_df_1000['score'] = final_gene_scores
final_gene_df_1000 = pd.DataFrame(data=final_gene_df_1000)

for i, ga in enumerate(ga_instances_1000):
    ga.save(f"ga_1000_{i}")
    
for i, df in enumerate(gene_df_gas_1000):
    df.to_csv(f"gene_df_1000_{i}.csv")
    
for i, df in enumerate(sol_df_gas_1000):
    df.to_csv(f"sol_df_1000_{i}.csv")
    
final_gene_df_1000.to_csv("final_gene_df_1000.csv")

# # %%

# gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes,
#                                             num_genes=num_genes,
#                                             num_generations=num_generations,
#                                             init_copies=init_copies,
#                                             mutation_probability=[0.5, 0],
#                                             set_genes=['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7'])

# # %%

# init_copies = 1

# gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes,
#                                             num_genes=num_genes, 
#                                             init_copies=init_copies,
#                                             num_generations=5, 
#                                             mutation_probability=[0.5, 0])

# # %% single most informative gene

# res_list = []
# for top_gene in tqdm(top_genes):
#     res_list.append(classify_cells.cross_val_classifier_single(meta_boot, exp_boot, cm_freqs, top_gene))
    
# fitness_list = []
# for res in res_list:
#     acc = np.mean(res["accs"])
#     sparsity = np.mean(res["sparsities"])
    
#     fitness_list.append((1/3)*acc + (2/3)*sparsity)
    
# res_df = {}
# res_df['gene'] = top_genes
# res_df['fitness'] = fitness_list
# res_df = pd.DataFrame(data=res_df)


gene_df = pd.read_csv("final_gene_df_1000.csv")

final_gene_df_filt = gene_df.nlargest(500, 'score')

top_genes = final_gene_df_filt['gene'].values


# number of genes in panel
num_genes = 24

# number of generations
num_generations = 100

# number of guaranteed copies of each gene in starter population
init_copies = 2

gene_df_gas_500 = []
sol_df_gas_500 = []
ga_instances_500 = []

for _ in range(10):

    meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
    exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
    freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
    
    exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=3)
    
    exp_super = gene_funcs.normalize_counts_to_median(exp_super)
    
    # bootstrap distribution from scRNAseq that matches MERFISH frequencies
    exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                        n=5000)
    
    exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)
    
    gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes, 
                                                num_generations=num_generations,
                                                init_copies=init_copies,
                                                num_genes=num_genes,
                                                mutation_probability=[0.5, 0],
                                                set_genes=['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7',
                                                           'Hoxb5', 'Ebf3', 'Pbx3', 'Nxph1',
                                                           'Arpp21', 'Zfhx4', 'Meis2', 'Rbms3'])
    
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
final_gene_df_500.to_csv("final_gene_df_500.csv")

final_gene_df_filt = final_gene_df_500.nlargest(200, 'score')

# %%

data, freqs = classify_cells.preprocess_data_splits(boots, freqs)

# %%

res = classify_cells.cross_val_classifier_splits(data, freqs, genes=['Phox2b', 'Slc17a6',
                                                                     'Slc6a5', 'Slc5a7'])

# %%

# number of genes in panel
num_genes = 24

# number of generations, default 100
num_generations = 100

# number of guaranteed copies of each gene in starter population, default 4
init_copies = 4

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
                                                init_copies=init_copies,
                                                num_generations=num_generations,
                                                num_genes=num_genes,
                                                mutation_probability=[0.5, 0],
                                                set_genes=['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7',
                                                           'Hoxb5', 'Ebf3', 'Pbx3', 'Nxph1',
                                                           'Arpp21', 'Zfhx4', 'Meis2', 'Rbms3'])
    
    gene_df_gas_200.append(gene_df_ga)
    sol_df_gas_200.append(sol_df_ga)
    ga_instances_200.append(ga_instance)

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
    
final_gene_df_200 = {}
final_gene_df_200['gene'] = tested_genes
final_gene_df_200['score'] = final_gene_scores
final_gene_df_200 = pd.DataFrame(data=final_gene_df_200)
final_gene_df_200.to_csv("final_gene_df_200.csv")
    
# %% final final 

gene_df_gas_final = []
sol_df_gas_final = []
ga_instances_final = []

top_genes = ["Zfhx3", "Hoxb5", "Meis2", "Nrxn3", "Ebf3", "Zfhx4", "Nxph1", "Pcdh7", 
             "Rbfox1", "Pbx3", "Pcp4", "Arpp21", "Gabra1", "Nxph4", "Pax2", "Syt1", 
             "Shox2", "Cdh13", "Syt2", "Nrg1", "Opcml", "Elavl2", "Ralyl", "Nrn1",
             "Cacna2d3", "Rbms3", "Car10", "Ntng1", "Rph3a", "Kcnip4", "Sorcs3",
             "Dpp10", "Atp2b4", "Cdh8", "Hcn1", "Lamp5", "Robo1", "Bcl11a", "Rab3c",
             "Cplx1", "Ahi1", "Gria3", "Cacna2d1", "Snca", "Kcna2", "Pax8", "Chrm3",
             "Scn1a", "Galnt13", "Cntn4", "Otp", "Olfm1", "Bend6", "Fndc5", "Atp2b2",
             "Vamp1", "Etl4", "Oprm1", "Necab1", "Slc30a3", "Vsnl1", "Scn8a"]

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
                                                num_genes=17,
                                                init_copies=4,
                                                num_generations=100, 
                                                mutation_probability=[0.5, 0],
                                                set_genes=['Phox2b', 'Slc32a1', 'Slc17a6'])
    
    gene_df_gas_final.append(gene_df_ga)
    sol_df_gas_final.append(sol_df_ga)
    ga_instances_final.append(ga_instance)

    
# %% collate results

tested_genes = []
for gene_df in gene_df_gas_final:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)

final_gene_scores = []
for gene in tested_genes:
    scores = []
    occurences = []
    for gene_df in gene_df_gas_final:
        if gene in gene_df['gene'].values:
            scores.append(gene_df['fitnesses'][gene_df['gene'] == gene].values[0])
            occurences.append(gene_df['occurences'][gene_df['gene'] == gene].values[0])
    final_gene_scores.append(np.average(scores, weights=occurences))
    
final_gene_df = {}
final_gene_df['gene'] = tested_genes
final_gene_df['score'] = final_gene_scores
final_gene_df = pd.DataFrame(data=final_gene_df)
final_gene_df.to_csv("final_gene_df_100.csv")

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

final_gene_df_filt = final_gene_df.nlargest(100, 'score')

top_genes = final_gene_df_filt['gene'].values

# %%

gene_df = pd.read_csv("final_gene_df_100.csv")

top_genes = gene_df['gene'].values

np.save("top_genes_100.npy", top_genes)



# %%

# number of genes in panel
num_genes = 24

# number of generations, default 100
num_generations = 100

# number of guaranteed copies of each gene in starter population, default 4
init_copies = 6

gene_df_gas_100 = []
sol_df_gas_100 = []
ga_instances_100 = []

for _ in range(10):

    meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
    exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
    freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
    
    exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=3)
    
    exp_super = gene_funcs.normalize_counts_to_median(exp_super)
    
    # bootstrap distribution from scRNAseq that matches MERFISH frequencies
    exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                        n=5000)
    
    exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)
    
    gene_df_ga, sol_df_ga, ga_instance = run_ga(meta_boot, exp_boot, cm_freqs, top_genes, 
                                                init_copies=init_copies,
                                                num_generations=num_generations,
                                                num_genes=num_genes,
                                                mutation_probability=[0.5, 0],
                                                set_genes=['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7',
                                                           'Hoxb5', 'Ebf3', 'Pbx3', 'Nxph1'])
    
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

for i, ga in enumerate(ga_instances_final):
    ga.save(f"ga_62_{i}")
    
for i, df in enumerate(gene_df_gas_final):
    df.to_csv(f"gene_df_62_{i}.csv")
    
for i, df in enumerate(sol_df_gas_final):
    df.to_csv(f"sol_df_62_{i}.csv")
    
final_gene_df.to_csv("final_gene_df_62.csv")
    
# %%

top_n = 1000



meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=3)

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

final_gene_df = pd.read_csv("final_gene_df_100_orig.csv")
final_gene_df = final_gene_df.nlargest(100, 'score')

inds = []
for gene in final_gene_df['gene'].values:
    inds.append(np.where(gene_df['gene'].values == gene)[0][0])

plt.hist(inds, bins=30)
plt.xlim(0, 2000)
plt.xlabel("Heuristic rank")
plt.ylabel("Frequency")
plt.title("Initial heuristic rank of top 100 final genes")

# %%

ABC_plot.plot_gene("Pcp4", meta_boot, exp_boot, level='cluster')

# %%

import random
import matplotlib.pyplot as plt 

# genes = ['Alcam','Arpp21','Ebf3','Hoxb5','Meis2','Negr1','Nxph1','Pax2','Pcp4',
#          'Phox2b','Rab3c','Robo1','Rph3a','Slc17a6','Slc5a7','Slc6a5','Syt1',
#          'Tenm2','Tshz2','Zfhx3']

genes = ['Phox2b', 'Slc17a6', 'Slc6a5', 'Slc5a7', 'Zfhx3', 'Meis2', 'Tenm2', 
         'Tshz2', 'Hoxb5', 'Ebf3', 'Ralyl', 'Rph3a', 'Syt1', 'Alcam', 'Arpp21', 'Pcp4']
# genes_list = [['Phox2b', 'Tenm2', 'Ralyl'], ['Tshz2', 'Alcam', 'Arpp21'], ['Zfhx3', 'Meis2', 'Pcp4']]

    
rand_genes = False

if rand_genes == True:
    all_genes = ABC_utils.load_gene("scRNAseq")
    
    genes = random.sample(list(all_genes), 16)

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
    idx_genes = np.array(idx_genes)[idx]
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_genes


R = np.corrcoef(exp_boot[:,gene_idx].T)
R, idx_genes = cluster_corr(R, genes)


fig, ax = plt.subplots()
mappable = ax.imshow(R, cmap='bwr')

plt.gca().set_xticks(range(len(idx_genes)), idx_genes, rotation=45)
plt.gca().set_yticks(range(len(idx_genes)), idx_genes, rotation=45)
mappable.set_clim(-1, 1)
cbar = plt.colorbar(mappable)
cbar.set_label("PCC", rotation=270, labelpad=15)

# %%

# optimized
# genes = ['Phox2b', 'Slc6a5', 'Slc17a6', 'Slc5a7',
#             'Hoxb5', 'Ebf3', 'Pbx3', 'Nxph1',
#             'Arpp21', 'Zfhx4', 'Meis2', 'Rbms3',
#             'Rbfox1', 'Grin3a', 'Bend6', 'Syt2',
#             'Tshz2', 'Tenm2', 'Kcna2', 'Pax2',
#             'Cntnap2', 'Pcdh7', 'Sdk1', 'Vamp1']
# HVGs
# genes = ['Fn1', 'Dbh', 'Ttn', 'Mafb', 'Mecom', 'Atp1a1', 'Kcnq4', 'Spp1',
#        'Gm32647', 'Calca', 'Slc6a2', 'Maf', 'Cd24a', 'Npb', 'Gm32004',
#        'Slc18a3', 'Clu', 'Pvalb', 'Chodl', 'Gm42418', 'AY036118',
#        'Slc5a7', 'Ttr', 'Slc18a2']

genes = ['Fn1', 'Dbh', 'Ttn', 'Mafb', 'Mecom', 'Atp1a1', 'Kcnq4', 'Spp1',
       'Gm32647', 'Calca', 'Slc6a2', 'Maf']
# all_genes = ABC_utils.load_gene("scRNAseq")

# gene_idx = []
# for gene in genes:
#     gene_idx.append(np.where(all_genes == gene)[0][0])
    
meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

splits = cell_funcs.K_fold_cells(meta)
supers = cell_funcs.boot_super_splits(exp, meta, splits, k=5)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
boots = cell_funcs.bootstrap_scRNAseq_splits(supers, freqs, n=5000)

data, freqs = classify_cells.preprocess_data_splits(boots, freqs)

final_labels, final_cm, all_accs, all_n_labels = iterative_reclustering.iterative_recluster_splits(data,
                                                                                                   freqs, 
                                                                                                   genes=genes, 
                                                                                                   acc_thresh=1)

hvgs_24_n_labels = all_n_labels
hvgs_24_accs = all_accs

hvgs_24_n_labels.append(1)
hvgs_24_accs.append(1)

# %%

import scanpy as sc
from anndata import AnnData, concat, read_h5ad

adata_us = read_h5ad(os.path.join(params.local_data_dir, "antIRN-PARN-upsampled"))

X = adata_us.X.copy()

std = adata_us.var['std']
std = std.values
for i in tqdm(range(X.shape[1])):
    X[:,i] = X[:,i]*std[i]
    
mean = adata_us.var['mean']
mean = mean.values
for i in tqdm(range(X.shape[1])):
    X[:,i] = X[:,i] + mean[i]
    
X = np.exp(X) - 1
X = np.round(X)
X = X.astype(np.uint16)

adata_us.X = X

sc.pp.highly_variable_genes(
    adata_us,
    flavor="seurat_v3",
    n_top_genes=16
)

hvgs = adata_us.var_names[adata_us.var['highly_variable']]


# %%

mix_targets = []

# motor neurons
mix_targets.append(('4601 HB Calcb Chol_1', '4603 HB Calcb Chol_3', '4604 HB Calcb Chol_4', '4605 HB Calcb Chol_5',
                    '4553 DMX VII Tbx20 Chol_1', '4554 DMX VII Tbx20 Chol_1', '4555 DMX VII Tbx20 Chol_1'))

# mossy fibers
mix_targets.append(('4198 PG-TRN-LRN Fat2 Glut_1', '4199 PG-TRN-LRN Fat2 Glut_1'))

labels = list(freqs.keys())
new_groups = cluster_recombination.manual_recluster(labels, mix_targets)
labels_mix = cluster_recombination.mix_labels(labels, new_groups)
clu_mapping = cluster_recombination.build_clu_mapping(labels, labels_mix)

# optimized
# genes = ['Phox2b', 'Slc17a6', 'Slc6a5', 'Slc5a7', 'Zfhx3', 'Hoxb5', 'Nxph1', 
#          'Nrg1', 'Ebf3', 'Pcp4', 'Scn1a', 'Celf2', 'Tenm2', 'Sdk1', 'Robo1',
#          'Pcdh7', 'Tshz2', 'Syt1', 'Pax2', 'Rbfox1', 'Ralyl', 'Alcam', 'Rph3a', 'Gria3']
# genes = ['Alcam','Arpp21','Ebf3','Hoxb5','Meis2','Negr1','Nxph1','Pax2','Pcp4',
#          'Phox2b','Rab3c','Robo1','Rph3a','Slc17a6','Slc5a7','Slc6a5','Syt1',
#          'Tenm2','Tshz2','Zfhx3']

# genes = ['Fn1', 'Dbh', 'Ttn', 'Mafb', 'Kcnq4', 'Spp1', 'Cntnap2', 'Slc6a2',
#        'Cd24a', 'Meg3', 'Gm32004', 'Pcdh9', 'Lsamp', 'Gm42418', 'Ttr',
#        'Malat1', 'mt-Co1', 'mt-Co2', 'mt-Atp6', 'mt-Co3']

genes = ['Fn1', 'Kcnq4', 'Spp1', 'Cntnap2', 'Slc6a2', 'Cd24a', 'Meg3',
       'Pcdh9', 'Lsamp', 'Gm42418', 'Ttr', 'Malat1', 'mt-Co1', 'mt-Co2',
       'mt-Atp6', 'mt-Co3']

# genes = ['Slc17a6', 'Tshz2', 'Phox2b', 'Ralyl', 'Rph3a', 'Pcp4', 'Meis2', 'Ebf3',
#          'Slc6a5', 'Alcam', 'Tenm2', 'Celf2', 'Slc5a7', 'Zfhx3', 'Syt1', 'Hoxb5']
    
meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

splits = cell_funcs.K_fold_cells(meta)
supers = cell_funcs.boot_super_splits(exp, meta, splits, k=3)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
boots = cell_funcs.bootstrap_scRNAseq_splits(supers, freqs, n=5000)

final_labels, final_cm, all_accs, all_n_labels, bits = cluster_recombination.iterative_recluster_splits(boots,
                                                                                                   freqs,
                                                                                                   clu_mapping=clu_mapping,
                                                                                                   genes=genes, 
                                                                                                   acc_thresh=0.99999)

opt_n_labels = all_n_labels
opt_accs = all_accs

opt_n_labels.append(1)
opt_accs.append(1)
bits.append(0)

# %%

fig, ax = plt.subplots()

ax.plot(opt_n_labels, opt_accs)
ax.set_ylim((0.89, 1.01))
ax.hlines(0.95, 0, 124, 'k', '--')

# %%

reclust_df = {}
reclust_df["n_labels"] = opt_n_labels
reclust_df["accs"] = opt_accs
reclust_df["bits"] = bits

reclust_df = pd.DataFrame(data=reclust_df)
reclust_df.to_csv("reclust_16_hvg.csv", index=False)

# %%

mix_targets = []

# motor neurons
mix_targets.append(('4601 HB Calcb Chol_1', '4603 HB Calcb Chol_3', '4604 HB Calcb Chol_4', '4605 HB Calcb Chol_5',
                    '4553 DMX VII Tbx20 Chol_1', '4554 DMX VII Tbx20 Chol_1', '4555 DMX VII Tbx20 Chol_1'))

# mossy fibers
mix_targets.append(('4198 PG-TRN-LRN Fat2 Glut_1', '4199 PG-TRN-LRN Fat2 Glut_1'))

labels = list(freqs.keys())
new_groups = cluster_recombination.manual_recluster(labels, mix_targets)
labels_mix = cluster_recombination.mix_labels(labels, new_groups)
clu_mapping = cluster_recombination.build_clu_mapping(labels, labels_mix)

import matplotlib.pyplot as plt
from scipy.stats import entropy

reclust_16_opt = pd.read_csv("reclust_16_opt.csv")
reclust_16_hvg = pd.read_csv("reclust_16_hvg.csv")

fig, ax = plt.subplots()

ax.plot(reclust_16_opt["n_labels"], reclust_16_opt["accs"], label='16 optimized genes')
ax.plot(reclust_16_hvg["n_labels"], reclust_16_hvg["accs"], label='16 highest variance genes')

fig.legend()
ax.set_xlabel("# cell types")
ax.set_ylabel("Accuracy")
ax.set_ylim((0, 1))
ax.set_xlim((1,124))

freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
cm_freqs = cell_funcs.freqs_to_cm_freqs(freqs, clu_mapping)
S = entropy(list(cm_freqs.values()), base=2)

def extract_bit(df, acc_thresh=0.95):
    
    mask = df["accs"] >= acc_thresh
    mask = mask.values
    mask = list(mask)
    idx = mask.index(True)

    bits = df["bits"].iloc[idx]
    return bits

bit_16_opt = extract_bit(reclust_16_opt)
bit_16_hvg = extract_bit(reclust_16_hvg)

fig, ax = plt.subplots()
ax.bar(['opt', 'hvg'], [bit_16_opt, bit_16_hvg])
ax.set_ylabel("Bits of information")
ax.axhline(S, 0, 1, c='k', ls='--')
ax.set_ylim((0, 6.5))

# %%

mix_targets = []

# motor neurons
mix_targets.append(('4601 HB Calcb Chol_1', '4603 HB Calcb Chol_3', '4604 HB Calcb Chol_4', '4605 HB Calcb Chol_5',
                    '4553 DMX VII Tbx20 Chol_1', '4554 DMX VII Tbx20 Chol_1', '4555 DMX VII Tbx20 Chol_1'))

# mossy fibers
mix_targets.append(('4198 PG-TRN-LRN Fat2 Glut_1', '4199 PG-TRN-LRN Fat2 Glut_1'))

labels = list(freqs.keys())
new_groups = cluster_recombination.manual_recluster(labels, mix_targets)
labels_mix = cluster_recombination.mix_labels(labels, new_groups)
clu_mapping = cluster_recombination.build_clu_mapping(labels, labels_mix)

import matplotlib.pyplot as plt
from scipy.stats import entropy

reclust_20_opt = pd.read_csv("reclust_20_opt.csv")
reclust_20_hvg = pd.read_csv("reclust_20_hvg.csv")

fig, ax = plt.subplots()

ax.plot(reclust_20_opt["n_labels"], reclust_20_opt["accs"], label='20 genes optimized')
ax.plot(reclust_20_hvg["n_labels"], reclust_20_hvg["accs"], label='20 most highly variable genes')

fig.legend()
ax.set_xlabel("# cell types")
ax.set_ylabel("Accuracy")
ax.set_ylim((0.5, 1.05))

freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
cm_freqs = cell_funcs.freqs_to_cm_freqs(freqs, clu_mapping)
S = entropy(list(cm_freqs.values()), base=2)

def extract_bit(df, acc_thresh=0.95):
    
    mask = df["accs"] >= acc_thresh
    mask = mask.values
    mask = list(mask)
    idx = mask.index(True)

    bits = df["bits"].iloc[idx]
    return bits

bit_20_opt = extract_bit(reclust_20_opt)
bit_20_hvg = extract_bit(reclust_20_hvg)

fig, ax = plt.subplots()
ax.bar(['opt', 'hvg'], [bit_20_opt, bit_20_hvg])
ax.set_ylabel("Bits of information")
ax.axhline(S, 0, 1, c='k', ls='--')
ax.set_ylim((0, 6.5))

fig, ax = plt.subplots()
bit_20_opt = (bit_20_opt/S)*100
bit_20_hvg = (bit_20_hvg/S)*100
ax.bar(['opt', 'hvg'], [bit_20_opt, bit_20_hvg])
ax.set_ylabel("Bits of information")
ax.set_ylim((0, 100))

# %%

mix_targets = []

# motor neurons
mix_targets.append(('4601 HB Calcb Chol_1', '4603 HB Calcb Chol_3', '4604 HB Calcb Chol_4', '4605 HB Calcb Chol_5',
                    '4553 DMX VII Tbx20 Chol_1', '4554 DMX VII Tbx20 Chol_1', '4555 DMX VII Tbx20 Chol_1'))

# mossy fibers
mix_targets.append(('4198 PG-TRN-LRN Fat2 Glut_1', '4199 PG-TRN-LRN Fat2 Glut_1'))

labels = list(freqs.keys())
new_groups = cluster_recombination.manual_recluster(labels, mix_targets)
labels_mix = cluster_recombination.mix_labels(labels, new_groups)
clu_mapping = cluster_recombination.build_clu_mapping(labels, labels_mix)

import matplotlib.pyplot as plt
from scipy.stats import entropy

reclust_4 = pd.read_csv("reclust_4.csv")
reclust_8 = pd.read_csv("reclust_8.csv")
reclust_12 = pd.read_csv("reclust_12.csv")
reclust_14 = pd.read_csv("reclust_14.csv")
reclust_16 = pd.read_csv("reclust_16.csv")
reclust_18 = pd.read_csv("reclust_18.csv")
reclust_20 = pd.read_csv("reclust_20.csv")
reclust_22 = pd.read_csv("reclust_22.csv")
reclust_24 = pd.read_csv("reclust_24.csv")

fig, ax = plt.subplots()

ax.plot(reclust_24["n_labels"], reclust_24["accs"], label='24 genes')
ax.plot(reclust_22["n_labels"], reclust_22["accs"], label='22 genes')
ax.plot(reclust_20["n_labels"], reclust_20["accs"], label='20 genes')
ax.plot(reclust_18["n_labels"], reclust_18["accs"], label='18 genes')
ax.plot(reclust_16["n_labels"], reclust_16["accs"], label='16 genes')
ax.plot(reclust_14["n_labels"], reclust_14["accs"], label='14 genes')
ax.plot(reclust_12["n_labels"], reclust_12["accs"], label='12 genes')
ax.plot(reclust_8["n_labels"], reclust_8["accs"], label='8 genes')
ax.plot(reclust_4["n_labels"], reclust_4["accs"], label='4 genes')

fig.legend()
ax.set_xlabel("# cell types")
ax.set_ylabel("Accuracy")
ax.set_ylim((0, 1))

freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))
cm_freqs = cell_funcs.freqs_to_cm_freqs(freqs, clu_mapping)
S = entropy(list(cm_freqs.values()), base=2)

def extract_bit(df, acc_thresh=0.95):
    
    mask = df["accs"] >= acc_thresh
    mask = mask.values
    mask = list(mask)
    idx = mask.index(True)

    bits = df["bits"].iloc[idx]
    return bits

bit_4 = extract_bit(reclust_4)
bit_8 = extract_bit(reclust_8)
bit_12 = extract_bit(reclust_12)
bit_14 = extract_bit(reclust_14)
bit_16 = extract_bit(reclust_16)
bit_18 = extract_bit(reclust_18)
bit_20 = extract_bit(reclust_20)
bit_22 = extract_bit(reclust_22)
bit_24 = extract_bit(reclust_24)

x = [4, 8, 12, 14, 16, 18, 20, 22, 24]
y = [bit_4, bit_8, bit_12, bit_14, bit_16, bit_18, bit_20, bit_22, bit_24]
y_norm = [(el/S)*100 for el in y]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel("# genes")
ax.set_ylabel("Bits of information")
ax.hlines(S, 0, 25)
ax.set_ylim((0, 7))

fig, ax = plt.subplots()
ax.scatter(x, y_norm)
ax.set_ylim((0, 100))
ax.set_xlabel("# genes")
ax.set_ylabel("% cell type information")




    
# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots()


ax.plot(hvgs_24_n_labels, hvgs_24_accs, label='HVGs')
ax.plot(opt_24_n_labels, opt_24_accs, label='optimized')
plt.xlabel("# cluster groups")
plt.ylabel("Accuracy")
plt.legend()
plt.title("12 genes, k=5 supercells")

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

# %%

dir1 = r"C:\Users\jpv88\Documents\GenePicker9001_data\2025-04-22 GA run final 62 Slc6a5"
dir2 = r"C:\Users\jpv88\Documents\GenePicker9001_data\2025-04-22 GA run final 62 Slc32a1"

gene_df_1 = pd.read_csv(os.path.join(dir1, "final_gene_df_62.csv"))
gene_df_2 = pd.read_csv(os.path.join(dir2, "final_gene_df_62.csv"))

gene_df_1.set_index("gene", inplace=True)
gene_df_2.set_index("gene", inplace=True)

gene_df_1.drop(columns="Unnamed: 0", inplace=True)
gene_df_2.drop(columns="Unnamed: 0", inplace=True)

gene_df = pd.concat((gene_df_1, gene_df_2), axis=1)
s = gene_df.mean(axis=1)
s.sort_values(ascending=False, inplace=True)

# %% relationship between occurences and score

dir1 = r"C:\Users\jpv88\Documents\GenePicker9001_data\2025-06-04 GA run"

gene_dfs = []
for i in range(10):
    gene_dfs.append(pd.read_csv(os.path.join(dir1, f"gene_df_1000_{i}.csv")))

tested_genes = []
for gene_df in gene_dfs:
    tested_genes.extend(gene_df['gene'].values)
    
tested_genes = np.unique(tested_genes)
    
final_gene_scores = {}
final_gene_occurences = {}
for gene in tested_genes:
    final_gene_scores[gene] = []
    final_gene_occurences[gene] = []
    
for gene in tested_genes:
    scores = []
    occurences = []
    for gene_df in gene_dfs:
        if gene in gene_df['gene'].values:
            
            final_gene_occurences[gene].append(gene_df['occurences'][gene_df['gene'] == gene].values[0])
            final_gene_scores[gene].append(gene_df['fitnesses'][gene_df['gene'] == gene].values[0])
            
for gene in tested_genes:
    final_gene_scores[gene] = np.average(final_gene_scores[gene], weights=final_gene_occurences[gene])
    
for gene in tested_genes:
    final_gene_occurences[gene] = np.sum(final_gene_occurences[gene])
    
final_gene_scores = np.array(list(final_gene_scores.values()))
final_gene_scores = (final_gene_scores - np.min(final_gene_scores)) / (np.max(final_gene_scores) - np.min(final_gene_scores))
    
plt.scatter(final_gene_scores, final_gene_occurences.values(), s=12)

# %%

t1000 = pd.read_csv("final_gene_df_100_top1000.csv")
t3000 = pd.read_csv("final_gene_df_100_top3000.csv")

# 50 genes are shared between the two conditions

# %%

top_n = 3000

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
for gene in t3000['gene'].values:
    inds.append(np.where(gene_df['gene'].values == gene)[0][0])

plt.hist(inds, bins=50)
plt.xlim(0, 3000)
plt.xlabel("Heuristic rank")
plt.ylabel("Frequency")
plt.title("Initial heuristic rank of top 100 final genes")

# %%

# optimized
genes = ['Nell1', 'Lrrtm4', 'Nrxn3', 'Hs6st3', 'Egfem1', 'Dpp10', 'Sema3e', 'Caln1',
 'Rbfox1', 'Timp2', 'Grin2a', 'Cadm2', 'Slc24a2', 'Sorcs3', 'Necab1', 'Cntnap5a',
 'Tenm2', 'March1', 'Pcdh9', 'Brinp3']
# genes = ["Penk", "Calb1", "Lamp5", "Rorb", "Ccdc80", "Pamr1", "Dkkl1", "Npnt",
#          "Lratd2", "Slco2a1", "Myl4", "Syt6", "Ctgf", "Slc32a1", "Pvalb", "Sst", "Vip"]

    
meta = pd.read_csv(os.path.join(params.local_data_dir, "MO-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "MO-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "MO-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

splits = cell_funcs.K_fold_cells(meta)
supers = cell_funcs.boot_super_splits(exp, meta, splits, k=3, boot_factor=0.5)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
boots = cell_funcs.bootstrap_scRNAseq_splits(supers, freqs, n=5000)

# %%

final_labels, final_cm, all_accs, all_n_labels, bits = cluster_recombination.iterative_recluster_splits(boots,
                                                                                                   freqs,
                                                                                                   genes=genes, 
                                                                                                   acc_thresh=0.95)

opt_n_labels = all_n_labels
opt_accs = all_accs

opt_n_labels.append(1)
opt_accs.append(1)
bits.append(0)

# %%

reclust_df = {}
reclust_df["n_labels"] = opt_n_labels
reclust_df["accs"] = opt_accs
reclust_df["bits"] = bits

reclust_df = pd.DataFrame(data=reclust_df)
reclust_df.to_csv("reclust_17_orig.csv", index=False)

# %%
import matplotlib.pyplot as plt
from scipy.stats import entropy

reclust_17 = pd.read_csv("reclust_17_orig.csv")
reclust_20 = pd.read_csv("reclust_20_opt.csv")

fig, ax = plt.subplots()

ax.plot(reclust_20["n_labels"], reclust_20["accs"], label='20 genes (optimized)')
ax.plot(reclust_17["n_labels"], reclust_17["accs"], label='17 genes (original)')

fig.legend()
ax.set_xlabel("# cell types")
ax.set_ylabel("Accuracy")
ax.set_ylim((0, 1))

freqs = pd.read_pickle(os.path.join(params.local_data_dir, "MO-MERFISH-freqs.pkl"))
S = entropy(list(freqs.values()), base=2)

def extract_bit(df, acc_thresh=0.95):
    
    mask = df["accs"] >= acc_thresh
    mask = mask.values
    mask = list(mask)
    idx = mask.index(True)

    bits = df["bits"].iloc[idx]
    return bits

bit_17 = extract_bit(reclust_17)
bit_20 = extract_bit(reclust_20)

x = [17, 20]
y = [bit_17, bit_20]
y_norm = [(el/S)*100 for el in y]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel("# genes")
ax.set_ylabel("Bits of information")
ax.hlines(S, 0, 25)
ax.set_ylim((0, 7))

fig, ax = plt.subplots()
ax.scatter(x, y_norm)
ax.set_ylim((0, 100))
ax.set_xlabel("# genes")
ax.set_ylabel("% cell type information")
            


