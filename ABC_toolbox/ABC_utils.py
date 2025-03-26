# -*- coding: utf-8 -*-
"""
Convenience functions
"""

import os
import sys

# obtain this file's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

# add parent directory to path
pardir = os.path.dirname(dname)
sys.path.append(pardir)

# import params from parent directory
import params

import numpy as np

# %%

load_dir = os.path.join(dname, "util_files")

def load_gene(dtype):
    
    match dtype:
        
        case "MERFISH":
            file = os.path.join(load_dir, "gene_MERFISH.npy")
            gene = np.load(file, allow_pickle=True)
            return gene
        
        case "scRNAseq":
            file = file = os.path.join(load_dir, "gene_scRNAseq.npy")
            gene = np.load(file, allow_pickle=True)
            return gene


