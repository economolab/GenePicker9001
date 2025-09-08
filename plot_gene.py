# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 16:52:53 2025

@author: jpv88
"""

import logging
import os
import pygad
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

import params

from ABC_toolbox import ABC_plot, ABC_utils, cell_funcs, classify_cells, gene_funcs, cluster_recombination
from gene_filter import filt_genes
from genetic_algo import run_ga

# %%

meta = pd.read_csv(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-meta.csv"), low_memory=False)
exp = np.load(os.path.join(params.local_data_dir, "antIRN-PARN-scRNAseq-raw.npy"))
freqs = pd.read_pickle(os.path.join(params.local_data_dir, "antIRN-PARN-MERFISH-freqs.pkl"))

exp = gene_funcs.normalize_counts_to_median(exp)

exp_super, meta_super = cell_funcs.boot_super(exp, meta, k=3, boot_factor=1)

exp_boot, meta_boot = cell_funcs.bootstrap_scRNAseq(meta_super, exp_super, freqs, 
                                                    n=10000)


# %%

ABC_plot.plot_gene("Rab3c", meta_boot, exp_boot, level='cluster')
