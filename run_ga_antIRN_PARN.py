# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:00:54 2025

@author: jpv88
"""

import os

import numpy as np
import pandas as pd

import params

from ABC_toolbox import cluster_recombination, gene_funcs
from genetic_algo import find_genes

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

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

find_genes(meta, exp, freqs, num_generations=1, num_iter=1, filt_order=[100], init_copies=1,
           rdr_N=10, var_E_N=2, skip_heur=True, clu_mapping=clu_mapping)