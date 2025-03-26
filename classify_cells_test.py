# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 20:09:34 2025

@author: jpv88
"""

import os

import numpy as np
import pandas as pd

import params

from ABC_toolbox import cell_funcs, classify_cells, gene_funcs

# %% load data

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=5)

exp_super = gene_funcs.normalize_counts_to_median(exp_super)

# bootstrap distribution from scRNAseq that matches MERFISH frequencies
exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=5000)

exp_boot, meta_boot, cm_freqs = classify_cells.preprocess_data(exp_boot, meta_boot, freqs)

# %%

genes = ["Phox2b", "Slc32a1", "Spp1", "Slc17a7", "Meis2", "Zfhx4", "Tfap2b"]

res = classify_cells.cross_val_classifier(meta_boot, exp_boot, cm_freqs, genes=genes)

